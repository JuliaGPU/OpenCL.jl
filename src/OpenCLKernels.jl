module OpenCLInterface

using ..OpenCL
using ..OpenCL: @device_override, method_table, kernel_convert, clfunction

import KernelInterface as KI

import SPIRVIntrinsics

import StaticArrays

import Adapt


## Back-end Definition

# export OpenCLBackend


# The platform is part of the backend's configuration. A backend works with the task's
# active device if that is on its platform. Otherwise, work for the backend (allocations,
# copies, compilation and launches) first activates the default device of its platform,
# as `KI.device!` would, so that the arrays it creates can be used afterwards.
Base.@kwdef struct OpenCLBackend <: KI.Backend
    platform::cl.Platform = cl.platform()
end

# the device that `b` works with
function backend_device(b::OpenCLBackend)
    cl.platform() == b.platform && return cl.device()
    dev = cl.default_device(b.platform)
    dev === nothing && throw(ArgumentError("OpenCL platform \"$(b.platform.name)\" has no devices"))
    return dev
end

# make the backend's device the task's active device
@inline function activate(b::OpenCLBackend)
    cl.platform() == b.platform || cl.platform!(b.platform)
    return
end

KI.versioninfo(io::IO, ::OpenCLBackend) = OpenCL.versioninfo(io)

function KI.allocate(b::OpenCLBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T
    activate(b)
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

# OpenCL.jl creates a context per device
context_device(ctx::cl.Context) = ctx == cl.context() ? cl.device() : only(ctx.devices)

function KI.get_backend(A::CLArray)
    ctx = OpenCL.context(A)
    ctx == cl.context() && return OpenCLBackend(cl.platform())
    return OpenCLBackend(context_device(ctx).platform)
end

function KI.synchronize(b::OpenCLBackend)
    activate(b)
    cl.finish(cl.queue(); blocking=false)
    return
end

## Device Selection

# devices are numbered consecutively within the backend's platform, in enumeration order

function KI.ndevices(b::OpenCLBackend)
    Int(cl.ndevices(b.platform))
end

function device_index(b::OpenCLBackend, dev::cl.Device)
    id = findfirst(==(dev), cl.devices(b.platform))
    id === nothing &&
        throw(ArgumentError("OpenCL device $(dev.name) is not on the backend's platform \"$(b.platform.name)\""))
    return id
end

KI.device(b::OpenCLBackend) = device_index(b, backend_device(b))

KI.device(b::OpenCLBackend, A::CLArray) = device_index(b, context_device(OpenCL.context(A)))

function KI.device!(b::OpenCLBackend, id::Int)
    0 < id <= KI.ndevices(b) || throw(ArgumentError("Device id $id out of bounds."))
    devs = cl.devices(b.platform)

    cl.device!(devs[id])
    return nothing
end

## Memory Operations

function KI.copyto!(b::OpenCLBackend, A, B)
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must have the same length, got $(length(A)) and $(length(B))"))
    activate(b)
    copyto!(A, B)
    return A
end

KI.unsafe_free!(A::CLArray) = OpenCL.unsafe_free!(A)


## Kernel Launch


KI.argconvert(::OpenCLBackend, arg) = kernel_convert(arg)

function KI.kernel_function(backend::OpenCLBackend, f::F, tt::TT=Tuple{}; name = nothing, kwargs...) where {F,TT}
    activate(backend)
    # on devices that support it, `clfunction` fixes the sub-group width to
    # `cl.sub_group_size(dev)`, as `KI.sub_group_size` promises
    kern = clfunction(f, tt; name, kwargs...)
    KI.Kernel{OpenCLBackend, typeof(kern)}(backend, kern)
end

# the context that a kernel was compiled for
function kernel_context(kernel::KI.Kernel{OpenCLBackend})
    ctx = Ref{cl.cl_context}()
    cl.clGetKernelInfo(kernel.kern.fun, cl.CL_KERNEL_CONTEXT, sizeof(cl.cl_context), ctx, C_NULL)
    return ctx[]
end

function kernel_device(kernel::KI.Kernel{OpenCLBackend})
    ctx = kernel_context(kernel)
    ctx == cl.context().id && return cl.device()
    return context_device(cl.Context(ctx; retain=true))
end

@noinline function throw_device_mismatch(kernel)
    throw(ArgumentError("Cannot launch a kernel compiled for $(kernel_device(kernel).name) on $(cl.device().name)"))
end

function KI.launch(kernel::KI.Kernel{OpenCLBackend}, groups::Dims{3}, items::Dims{3}, args::Vararg{Any, N}; kwargs...) where {N}
    activate(kernel.backend)
    kernel_context(kernel) == cl.context().id || throw_device_mismatch(kernel)
    kernel.kern(args...; local_size = items, global_size = items .* groups, kwargs...)
    return
end

function KI.max_work_group_size(kernel::KI.Kernel{OpenCLBackend})::Int
    wginfo = cl.work_group_info(kernel.kern.fun, kernel_device(kernel))
    Int(wginfo.size)
end


## Device Properties

# querying the device allocates, so cache what launches and kernels need. the cache is
# keyed on the device, because the task-local device can be switched.
const DeviceProperties = @NamedTuple{
    max_work_group_size::Int, max_work_group_dims::NTuple{3, Int}, compute_units::Int,
    float64::Bool, float16::Bool, unified::Bool,
    # 0 if the device doesn't support sub-groups of a fixed width
    sub_group_size::Int, sub_group_shuffle::Bool,
}
function device_properties(dev::cl.Device)
    cache = get!(task_local_storage(), :CLDeviceProperties) do
        Dict{cl.Device, DeviceProperties}()
    end::Dict{cl.Device, DeviceProperties}
    return get!(cache, dev) do
        sizes = dev.max_work_item_size
        extensions = dev.extensions
        # the sub-group width is only fixed for kernels that request it, which `clfunction`
        # does for devices with `cl_intel_required_subgroup_size`
        fixed_sub_groups = cl.sub_groups_supported(dev) &&
                           "cl_intel_required_subgroup_size" in extensions
        (; max_work_group_size = Int(dev.max_work_group_size),
           max_work_group_dims = ntuple(d -> d <= length(sizes) ? Int(sizes[d]) : 1, 3),
           compute_units = Int(dev.max_compute_units),
           float64 = "cl_khr_fp64" in extensions,
           float16 = "cl_khr_fp16" in extensions,
           unified = cl.default_memory_backend(dev; unified=true) !== nothing,
           sub_group_size = fixed_sub_groups ? cl.sub_group_size(dev) : 0,
           sub_group_shuffle = fixed_sub_groups && "cl_khr_subgroup_shuffle" in extensions)
    end
end

device_properties(b::OpenCLBackend) = device_properties(backend_device(b))

KI.max_work_group_size(b::OpenCLBackend)::Int = device_properties(b).max_work_group_size
KI.max_work_group_dims(b::OpenCLBackend)::NTuple{3, Int} = device_properties(b).max_work_group_dims
# OpenCL doesn't limit the number of work-groups, only the global size (to `size_t`)
function KI.max_num_groups(b::OpenCLBackend)::NTuple{3, Int}
    return typemax(Int) .÷ KI.max_work_group_dims(b)
end
KI.multiprocessor_count(b::OpenCLBackend)::Int = device_properties(b).compute_units

KI.supports_float64(b::OpenCLBackend) = device_properties(b).float64
KI.supports_unified(b::OpenCLBackend) = device_properties(b).unified
# 32-bit integer atomics are core OpenCL; float atomics fall back to compare-and-swap
KI.supports_atomics(::OpenCLBackend) = true

KI.supports_subgroups(b::OpenCLBackend) = device_properties(b).sub_group_size > 0
KI.sub_group_size(b::OpenCLBackend)::Int = device_properties(b).sub_group_size
function KI.supports_shuffle(b::OpenCLBackend, ::Type{T}) where {T}
    props = device_properties(b)
    props.sub_group_shuffle || return false
    T in SPIRVIntrinsics.gentypes || return false
    T === Float64 && return props.float64
    T === Float16 && return props.float16
    return true
end

## Indexing Functions
## COV_EXCL_START

# computed with `% T`, which unlike `T(x)` has no error path. KernelInterface derives the
# global queries from these.

@device_override @inline function KI.get_local_id(::Type{T}) where {T}
    return (; x = get_local_id(1) % T, y = get_local_id(2) % T, z = get_local_id(3) % T)
end

@device_override @inline function KI.get_group_id(::Type{T}) where {T}
    return (; x = get_group_id(1) % T, y = get_group_id(2) % T, z = get_group_id(3) % T)
end

@device_override @inline function KI.get_local_size(::Type{T}) where {T}
    return (; x = get_local_size(1) % T, y = get_local_size(2) % T, z = get_local_size(3) % T)
end

@device_override @inline function KI.get_num_groups(::Type{T}) where {T}
    return (; x = get_num_groups(1) % T, y = get_num_groups(2) % T, z = get_num_groups(3) % T)
end

# OpenCL's sub-group queries already have KernelInterface's semantics: the last sub-group
# of a work-group can be partial, and `get_sub_group_size` counts the work-items present

@device_override KI.get_sub_group_size(::Type{T}) where {T} = get_sub_group_size() % T

@device_override KI.get_max_sub_group_size(::Type{T}) where {T} = get_max_sub_group_size() % T

@device_override KI.get_num_sub_groups(::Type{T}) where {T} = get_num_sub_groups() % T

@device_override KI.get_sub_group_id(::Type{T}) where {T} = get_sub_group_id() % T

@device_override KI.get_sub_group_local_id(::Type{T}) where {T} = get_sub_group_local_id() % T

## Shared and Scratch Memory

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ptr = OpenCL.emit_localmemory(T, Val(prod(Dims)))
    CLDeviceArray(Dims, ptr)
end

## Synchronization and Printing

@device_override @inline function KI.barrier()
    work_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
end

@device_override @inline function KI.sub_group_barrier()
    sub_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
end

# out-of-range source lanes give an undefined value, as KernelInterface allows
@device_override function KI.shfl_down(val::T, offset::Integer) where T
    sub_group_shuffle(val, get_sub_group_local_id() % UInt32 + offset % UInt32)
end

@device_override @inline function KI._print(args...)
    OpenCL._print(args...)
end
## COV_EXCL_STOP

end
