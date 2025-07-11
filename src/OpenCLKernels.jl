module OpenCLKernels

using ..OpenCL
using ..OpenCL: @device_override, method_table

import KernelAbstractions as KA

import StaticArrays

import Adapt


## Back-end Definition

export OpenCLBackend

struct OpenCLBackend <: KA.GPU
end

KA.allocate(::OpenCLBackend, ::Type{T}, dims::Tuple) where T = CLArray{T}(undef, dims)
KA.zeros(::OpenCLBackend, ::Type{T}, dims::Tuple) where T = OpenCL.zeros(T, dims)
KA.ones(::OpenCLBackend, ::Type{T}, dims::Tuple) where T = OpenCL.ones(T, dims)

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


## Indexing Functions

@device_override @inline function KA.__index_Local_Linear(ctx)
    return get_local_id(1)
end

@device_override @inline function KA.__index_Group_Linear(ctx)
    return get_group_id(1)
end

@device_override @inline function KA.__index_Global_Linear(ctx)
    return get_global_id(1)
end

@device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[get_local_id(1)]
end

@device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[get_group_id(1)]
end

@device_override @inline function KA.__index_Global_Cartesian(ctx)
    return @inbounds KA.expand(KA.__iterspace(ctx), get_group_id(1), get_local_id(1))
end

@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), get_group_id(1), get_local_id(1))
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end


## Shared and Scratch Memory

@device_override @inline function KA.SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    ptr = OpenCL.emit_localmemory(T, Val(prod(Dims)))
    CLDeviceArray(Dims, ptr)
end

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end


## Synchronization and Printing

@device_override @inline function KA.__synchronize()
    work_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
end

@device_override @inline function KA.__print(args...)
    OpenCL._print(args...)
end


## Other

KA.argconvert(::KA.Kernel{OpenCLBackend}, arg) = OpenCL.kernel_convert(arg)

end
