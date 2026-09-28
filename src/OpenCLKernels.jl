module OpenCLInterface

using ..OpenCL
using ..OpenCL: @device_override, method_table, kernel_convert, clfunction

import KernelInterface as KI

import SPIRVIntrinsics

import StaticArrays

import Adapt


## Back-end Definition

# export OpenCLBackend


Base.@kwdef struct OpenCLBackend <: KI.Backend
    platform::cl.Platform = cl.platform()
end

@noinline function platform_mismatch_warning(expected::cl.Platform, active::cl.Platform)
    @warn "OpenCLBackend platform \"$(expected.name)\" is not the active platform \"$(active.name)\""
    return nothing
end

KI.versioninfo(io::IO, ::OpenCLBackend) = OpenCL.versioninfo(io)

function KI.allocate(b::OpenCLBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T
    b.platform === cl.platform() || platform_mismatch_warning(b.platform, cl.platform())
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

KI.get_backend(::CLArray) = OpenCLBackend()
# TODO should be non-blocking
KI.synchronize(::OpenCLBackend) = cl.finish(cl.queue())

## Device Selection

# devices are numbered consecutively within the backend's platform, in enumeration order

function KI.ndevices(b::OpenCLBackend)
    Int(cl.ndevices(b.platform))
end

function KI.device(b::OpenCLBackend)
    current = cl.device()
    for (i, d) in enumerate(cl.devices(b.platform))
        d == current && return i
    end
    error("Active OpenCL device $current not found in the OpenCLBackend's platform \"$(b.platform.name)\".")
end

function KI.device!(b::OpenCLBackend, id::Int)
    0 < id <= KI.ndevices(b) || throw(ArgumentError("Device id $id out of bounds."))
    devs = cl.devices(b.platform)

    cl.device!(devs[id])
    return nothing
end

## Memory Operations

function KI.copyto!(::OpenCLBackend, A, B)
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must have the same length, got $(length(A)) and $(length(B))"))
    copyto!(A, B)
    return A
end

KI.unsafe_free!(A::CLArray) = OpenCL.unsafe_free!(A)


## Kernel Launch


KI.argconvert(::OpenCLBackend, arg) = kernel_convert(arg)

function KI.kernel_function(backend::OpenCLBackend, f::F, tt::TT=Tuple{}; name = nothing, kwargs...) where {F,TT}
    # on devices that support it, `clfunction` fixes the sub-group width to
    # `cl.sub_group_size(dev)`, as `KI.sub_group_size` promises
    kern = clfunction(f, tt; name, kwargs...)
    KI.Kernel{OpenCLBackend, typeof(kern)}(backend, kern)
end

function KI.launch(kernel::KI.Kernel{OpenCLBackend}, groups::Dims{3}, items::Dims{3}, args...; kwargs...)
    kernel.backend.platform === cl.platform() || platform_mismatch_warning(kernel.backend.platform, cl.platform())
    kernel.kern(args...; local_size = items, global_size = items .* groups, kwargs...)
    return
end

function KI.max_work_group_size(kernel::KI.Kernel{OpenCLBackend})::Int
    wginfo = cl.work_group_info(kernel.kern.fun, cl.device())
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
function device_properties(dev::cl.Device = cl.device())
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

KI.max_work_group_size(::OpenCLBackend)::Int = device_properties().max_work_group_size
KI.max_work_group_dims(::OpenCLBackend)::NTuple{3, Int} = device_properties().max_work_group_dims
# OpenCL doesn't limit the number of work-groups, only the global size (to `size_t`)
function KI.max_num_groups(b::OpenCLBackend)::NTuple{3, Int}
    return typemax(Int) .÷ KI.max_work_group_dims(b)
end
KI.multiprocessor_count(::OpenCLBackend)::Int = device_properties().compute_units

KI.supports_float64(::OpenCLBackend) = device_properties().float64
KI.supports_unified(::OpenCLBackend) = device_properties().unified
# 32-bit integer atomics are core OpenCL; float atomics fall back to compare-and-swap
KI.supports_atomics(::OpenCLBackend) = true

KI.supports_subgroups(::OpenCLBackend) = device_properties().sub_group_size > 0
KI.sub_group_size(::OpenCLBackend)::Int = device_properties().sub_group_size
function KI.supports_shuffle(::OpenCLBackend, ::Type{T}) where {T}
    props = device_properties()
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
