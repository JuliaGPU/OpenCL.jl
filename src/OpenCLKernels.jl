module OpenCLKernels

using ..OpenCL
using ..OpenCL: @device_override, method_table

import KernelAbstractions as KA

import StaticArrays

import Adapt


## Back-end Definition

export OpenCLBackend

struct OpenCLBackend <: KA.GPU
    platform::cl.Platform

    function OpenCLBackend(; platform=cl.platform())
        cl.platform!(platform)
        new(platform)
    end
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
KA.supports_float64(::OpenCLBackend) = in("cl_khr_fp64", cl.device().extensions)

Adapt.adapt_storage(::OpenCLBackend, a::Array) = Adapt.adapt(CLArray, a)
Adapt.adapt_storage(::OpenCLBackend, a::CLArray) = a
Adapt.adapt_storage(::KA.CPU, a::CLArray) = convert(Array, a)

# `@Const` applies `constify` inside the kernel, where arguments have already been
# converted to device arrays, so the rule has to be registered for `CLDeviceArray`
# rather than for `CLArray`.
Adapt.adapt_storage(::KA.ConstAdaptor, a::CLDeviceArray) = Base.Experimental.Const(a)

## Device Selection

# devices are numbered consecutively across all platforms, in enumeration order

function KA.ndevices(b::OpenCLBackend)
    length(cl.devices(b.platform))
end

function KA.device(b::OpenCLBackend)
    current = cl.device()
    for (i, d) in enumerate(cl.devices(b.platform))
        d == current && return i
    end
    error("Active OpenCL device $current not found in the current OpenCL platform \"$(b.platform.name)\".")
end

function KA.device!(b::OpenCLBackend, id::Int)
    devs = cl.devices(b.platform)
    0 < id <= length(devs) || throw(ArgumentError("Device id $id out of bounds."))

    cl.device!(devs[id])
    return nothing
end

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
    #return get_global_id(1)    # JuliaGPU/OpenCL.jl#346
    I = KA.__index_Global_Cartesian(ctx)
    @inbounds LinearIndices(KA.__ndrange(ctx))[I]
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
        I = KA.__index_Global_Cartesian(ctx)
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
