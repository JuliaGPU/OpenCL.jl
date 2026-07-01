module OpenCLKernels

using ..OpenCL
using ..OpenCL: @device_override, method_table, kernel_convert, clfunction

import KernelAbstractions as KA
import KernelAbstractions.KernelIntrinsics as KI

import StaticArrays

import Adapt


## Back-end Definition

export OpenCLBackend

struct OpenCLBackend <: KA.GPU
end

function KA.allocate(::OpenCLBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T
    if unified
        memory_backend = cl.unified_memory_backend()
        if memory_backend === cl.USMBackend()
            return CLArray{T, length(dims), cl.UnifiedSharedMemory}(undef, dims)
        elseif memory_backend === cl.SVMBackend()
            return CLArray{T, length(dims), cl.SharedVirtualMemory}(undef, dims)
        else
            throw(ArgumentError("Unified memory not supported"))
        end
    else
        return CLArray{T}(undef, dims)
    end
end

KA.supports_unified(::OpenCLBackend) = cl.default_memory_backend(cl.device(); unified=true) !== nothing

KA.get_backend(::CLArray) = OpenCLBackend()
# TODO should be non-blocking
KA.synchronize(::OpenCLBackend) = cl.finish(cl.queue())
KA.supports_float64(::OpenCLBackend) = false  # TODO: Check if this is device dependent

Adapt.adapt_storage(::OpenCLBackend, a::Array) = Adapt.adapt(CLArray, a)
Adapt.adapt_storage(::OpenCLBackend, a::CLArray) = a
Adapt.adapt_storage(::KA.CPU, a::CLArray) = convert(Array, a)


## Memory Operations

function KA.copyto!(::OpenCLBackend, A, B)
    copyto!(A, B)
    # TODO: Address device to host copies in jl being synchronizing
end


## Kernel Launch

function KA.mkcontext(kernel::KA.Kernel{OpenCLBackend}, _ndrange, iterspace)
    KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
end
function KA.mkcontext(kernel::KA.Kernel{OpenCLBackend}, I, _ndrange, iterspace,
                      ::Dynamic) where Dynamic
    KA.CompilerMetadata{KA.ndrange(kernel), Dynamic}(I, _ndrange, iterspace)
end

function KA.launch_config(kernel::KA.Kernel{OpenCLBackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    # partition checked that the ndrange's agreed
    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KA.workgroupsize(kernel) <: KA.DynamicSize &&
        workgroupsize === nothing
        # use ndrange as preliminary workgroupsize for autotuning
        KA.partition(kernel, ndrange, ndrange)
    else
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

function (obj::KA.Kernel{OpenCLBackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ndrange, workgroupsize, iterspace, dynamic =
        KA.launch_config(obj, ndrange, workgroupsize)

    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    kernel = @opencl launch=false obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        wg_info = cl.work_group_info(kernel.fun, cl.device())
        wg_size_nd = threads_to_workgroupsize(wg_info.size, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, wg_size_nd)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    groups = length(KA.blocks(iterspace))
    items = length(KA.workitems(iterspace))

    if groups == 0
        return nothing
    end

    # Launch kernel
    global_size = groups * items
    local_size = items
    kernel(ctx, args...; global_size, local_size)

    return nothing
end

KI.argconvert(::OpenCLBackend, arg) = kernel_convert(arg)

function KI.kernel_function(::OpenCLBackend, f::F, tt::TT=Tuple{}; name = nothing, kwargs...) where {F,TT}
    kern = clfunction(f, tt; name, kwargs...)
    KI.Kernel{OpenCLBackend, typeof(kern)}(OpenCLBackend(), kern)
end

function (obj::KI.Kernel{OpenCLBackend})(args...; numworkgroups = 1, workgroupsize = 1)
    KI.check_launch_args(numworkgroups, workgroupsize)

    local_size = (workgroupsize..., ntuple(_ -> 1, 3 - length(workgroupsize))...)

    numworkgroups = (numworkgroups..., ntuple(_ -> 1, 3 - length(numworkgroups))...)
    global_size = local_size .* numworkgroups

    obj.kern(args...; local_size, global_size)
    return nothing
end


function KI.kernel_max_work_group_size(kernel::KI.Kernel{<:OpenCLBackend}; max_work_items::Int=typemax(Int))::Int
    wginfo = cl.work_group_info(kernel.kern.fun, cl.device())
    Int(min(wginfo.size, max_work_items))
end
function KI.max_work_group_size(::OpenCLBackend)::Int
    Int(cl.device().max_work_group_size)
end
function KI.multiprocessor_count(::OpenCLBackend)::Int
    Int(cl.device().max_compute_units)
end

## Indexing Functions
## COV_EXCL_START

@device_override @inline function KI.get_local_id()
    return (; x = Int(get_local_id(1)), y = Int(get_local_id(2)), z = Int(get_local_id(3)))
end

@device_override @inline function KI.get_group_id()
    return (; x = Int(get_group_id(1)), y = Int(get_group_id(2)), z = Int(get_group_id(3)))
end

@device_override @inline function KI.get_global_id()
    return (; x = Int(get_global_id(1)), y = Int(get_global_id(2)), z = Int(get_global_id(3)))
end

@device_override @inline function KI.get_local_size()
    return (; x = Int(get_local_size(1)), y = Int(get_local_size(2)), z = Int(get_local_size(3)))
end

@device_override @inline function KI.get_num_groups()
    return (; x = Int(get_num_groups(1)), y = Int(get_num_groups(2)), z = Int(get_num_groups(3)))
end

@device_override @inline function KI.get_global_size()
    return (; x = Int(get_global_size(1)), y = Int(get_global_size(2)), z = Int(get_global_size(3)))
end

@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = KA.__index_Global_Cartesian(ctx)
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end


## Shared and Scratch Memory

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ptr = OpenCL.emit_localmemory(T, Val(prod(Dims)))
    CLDeviceArray(Dims, ptr)
end

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end


## Synchronization and Printing

@device_override @inline function KI.barrier()
    work_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
end

@device_override @inline function KI._print(args...)
    OpenCL._print(args...)
end
## COV_EXCL_STOP

## Other

KA.argconvert(::KA.Kernel{OpenCLBackend}, arg) = OpenCL.kernel_convert(arg)

end
