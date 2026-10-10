module OpenCLKernels

using ..OpenCL
using ..OpenCL: @device_override, method_table, kernel_convert, clfunction, has_feature

import KernelInterface as KI

import Adapt


## Back-end Definition

export OpenCLBackend

"""
    OpenCLBackend(; platform=cl.platform())

KernelInterface back end for the OpenCL devices of `platform`.

A backend works with the task's active device if that is on its platform. Otherwise, work
for the backend (allocations, copies, compilation and launches) first activates the default
device of its platform, as `KernelInterface.device!` would, so that the arrays it creates
can be used afterwards.

# Sub-groups

KernelInterface's sub-groups need a width that is the same for every kernel. For now, they
are only supported on devices with `cl_intel_required_subgroup_size`, through which kernels
request the width that `KernelInterface.sub_group_size` reports (at most 64). The shuffles
also need `cl_khr_subgroup_shuffle`.

OpenCL doesn't specify which work-items form a sub-group, so
`KernelInterface.supports_linear_subgroups` is only `true` on runtimes known to form them
from consecutive work-items: PoCL and Intel's CPU and GPU runtimes. On PoCL, sub-group operations are
work-group barriers, so `KernelInterface.supports_independent_subgroups` is `false`: all
sub-groups of a work-group have to execute the same sub-group operations, in the same order.
It is also `false` on Intel's CPU runtime, which loses the writes after a
`sub_group_barrier` that other sub-groups returned before (intel/llvm#23449).
"""
Base.@kwdef struct OpenCLBackend <: KI.Backend
    platform::cl.Platform = cl.platform()
end

KI.versioninfo(io::IO, ::OpenCLBackend) = OpenCL.versioninfo(io)

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
    OpenCL.synchronize()
    return
end

# queues are task-local, so work is ordered across tasks with a marker event on the
# recording task's queue
function KI.record_event(b::OpenCLBackend)
    activate(b)
    event = cl.enqueue_marker_with_wait_list(cl.AbstractEvent[])
    # the waiting queue only makes progress if this one is submitted
    cl.flush(cl.queue())
    return event
end

function event_context(event::cl.Event)
    ctx = Ref{cl.cl_context}()
    cl.clGetEventInfo(event, cl.CL_EVENT_CONTEXT, sizeof(cl.cl_context), ctx, C_NULL)
    return ctx[]
end

function KI.wait_event(b::OpenCLBackend, event::cl.Event)
    activate(b)
    # the event has to stay alive until the driver has retained it
    GC.@preserve event begin
        if event_context(event) == cl.context().id
            cl.enqueue_barrier_with_wait_list(cl.AbstractEvent[event])
        else
            # XXX: queues can only wait for events of their own context, and OpenCL.jl
            #      creates a context per device, so wait for other devices on the host.
            wait(event)
        end
    end
    return
end

function Adapt.adapt_storage(b::OpenCLBackend, a::Array)
    activate(b)
    return Adapt.adapt(CLArray, a)
end
Adapt.adapt_storage(::OpenCLBackend, a::CLArray) = a


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

# `f` is the host-side callable: the kernel keeps it as its `source`, which is converted
# again at every launch, because the converted callable only holds pointers to the arrays
# it captures
function KI.kernel_function(backend::OpenCLBackend, f::F, tt::TT=Tuple{}; name = nothing, kwargs...) where {F,TT}
    activate(backend)
    check_sub_group_size(backend, kwargs)
    kern = GC.@preserve f clfunction(kernel_convert(f), tt; source=f, name, kwargs...)
    KI.Kernel{OpenCLBackend, typeof(kern)}(backend, kern)
end

# kernels have to execute with the sub-group width that `KI.sub_group_size` reports
function check_sub_group_size(backend::OpenCLBackend, kwargs)
    haskey(kwargs, :sub_group_size) && KI.supports_subgroups(backend) || return
    width = KI.sub_group_size(backend)
    kwargs[:sub_group_size] == width ||
        throw(ArgumentError("KernelInterface kernels execute with sub-group width $width, got `sub_group_size=$(kwargs[:sub_group_size])`"))
    return
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

@noinline function throw_geometry_keyword()
    throw(ArgumentError("KernelInterface kernels take `numgroups`, `workgroupsize` or `ndrange`, not `global_size` or `local_size`"))
end

# passes the arguments on as a tuple, like calling the `HostKernel` does
function KI.launch(kernel::KI.Kernel{OpenCLBackend}, groups::Dims{3}, items::Dims{3},
                   args::Tuple; kwargs...)
    # KernelInterface has validated the launch geometry
    if haskey(kwargs, :global_size) || haskey(kwargs, :local_size)
        throw_geometry_keyword()
    end
    activate(kernel.backend)
    kernel_context(kernel) == cl.context().id || throw_device_mismatch(kernel)
    OpenCL.launch_tuple(kernel.kern, args; local_size = items, global_size = items .* groups,
                        kwargs...)
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
    float64::Bool, unified::Bool,
    # 0 if the device doesn't support sub-groups of a fixed width
    sub_group_size::Int, shuffle::Bool, float16::Bool,
    linear_sub_groups::Bool, independent_sub_groups::Bool,
}
function device_properties(dev::cl.Device)
    cache = get!(task_local_storage(), :CLDeviceProperties) do
        Dict{cl.Device, DeviceProperties}()
    end::Dict{cl.Device, DeviceProperties}
    return get!(cache, dev) do
        sizes = dev.max_work_item_size
        # KernelInterface needs a fixed sub-group width, which for now means one that kernels
        # request with `cl_intel_required_subgroup_size` (as `clfunction` does). Wider
        # sub-groups than 64 aren't reported, since `KI.sub_group_ballot` has a bit per lane.
        sub_group_size = 0
        if cl.sub_groups_supported(dev) && "cl_intel_required_subgroup_size" in dev.extensions
            width = cl.sub_group_size(dev)
            width <= 64 && (sub_group_size = width)
        end
        pocl = occursin("pocl", dev.platform.vendor)
        (; max_work_group_size = Int(dev.max_work_group_size),
           max_work_group_dims = ntuple(d -> d <= length(sizes) ? Int(sizes[d]) : 1, 3),
           compute_units = Int(dev.max_compute_units),
           float64 = "cl_khr_fp64" in dev.extensions,
           unified = cl.default_memory_backend(dev; unified=true) !== nothing,
           sub_group_size,
           shuffle = sub_group_size > 0 && "cl_khr_subgroup_shuffle" in dev.extensions,
           float16 = "cl_khr_fp16" in dev.extensions,
           # OpenCL leaves the layout to the implementation: only report it for the
           # runtimes known to form sub-groups from consecutive work-items
           linear_sub_groups = sub_group_size > 0 && (pocl || linear_sub_group_runtime(dev)),
           # PoCL executes sub-group operations as work-group barriers, and Intel's CPU
           # runtime loses the writes after a `sub_group_barrier` that other sub-groups
           # returned before (intel/llvm#23449)
           independent_sub_groups = sub_group_size > 0 && !pocl && !intel_cpu_runtime(dev))
    end
end

# Intel's CPU and GPU runtimes form sub-groups from consecutive work-items in the x
# dimension, as KernelInterface's layout tests check
function linear_sub_group_runtime(dev::cl.Device)
    return occursin("Intel", dev.platform.vendor) && dev.device_type in (:cpu, :gpu)
end

intel_cpu_runtime(dev::cl.Device) = occursin("Intel", dev.platform.vendor) && dev.device_type === :cpu

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
KI.supports_linear_subgroups(b::OpenCLBackend) = device_properties(b).linear_sub_groups
KI.supports_independent_subgroups(b::OpenCLBackend) = device_properties(b).independent_sub_groups

# The types `sub_group_shuffle` supports; other types are shuffled as words or field by
# field. `Float16` and `Float64` are reported as unsupported on devices without `cl_khr_fp16`
# or `cl_khr_fp64`: a kernel can't hold such values there, even just to move them, since
# LLVM turns a load of an integer that is reinterpreted as a float into a load of the float.
const ShuffleTypes = Union{OpenCL.SPIRVIntrinsics.gentypes...}

function KI.supports_shuffle(b::OpenCLBackend, ::Type{T}) where {T <: ShuffleTypes}
    props = device_properties(b)
    props.shuffle || return false
    T === Float64 && return props.float64
    T === Float16 && return props.float16
    return true
end


## Indexing Functions

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


## Shared Memory

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

# `cl_khr_subgroup_shuffle`
@device_override KI.shfl(val::T, lane::Integer) where {T <: ShuffleTypes} =
    sub_group_shuffle(val, lane)

# past the sub-group width, `shfl_down` and `shfl_up` return the work-item's own value, which
# `sub_group_shuffle` (like SPIR-V's `OpGroupNonUniformShuffleDown`) leaves undefined
@device_override function KI.shfl_down(val::T, offset::Integer) where {T <: ShuffleTypes}
    lane = get_sub_group_local_id()
    # compared before adding, so that large offsets don't overflow
    inside = offset <= get_max_sub_group_size() - lane
    return sub_group_shuffle(val, ifelse(inside, lane + offset, lane))
end

@device_override function KI.shfl_up(val::T, offset::Integer) where {T <: ShuffleTypes}
    lane = get_sub_group_local_id()
    return sub_group_shuffle(val, ifelse(lane > offset, lane - offset, lane))
end

@device_override KI.shfl_xor(val::T, mask::Integer) where {T <: ShuffleTypes} =
    sub_group_shuffle_xor(val, mask)

# `cl_khr_subgroups`
@device_override KI.sub_group_any(pred::Bool) = sub_group_any(pred)

@device_override KI.sub_group_all(pred::Bool) = sub_group_all(pred)

@device_override function KI.sub_group_ballot(pred::Bool)
    if has_feature(:subgroup_ballot)
        mask = sub_group_ballot(pred)
        return UInt64(mask[1].value) | (UInt64(mask[2].value) << 32)
    else
        return reduction_ballot(pred)
    end
end

# The ballot without `cl_khr_subgroup_ballot`: add up a bit per lane, with 32-bit reductions,
# which every device with sub-groups supports. The width is at most 64 (see
# `device_properties`).
@inline function reduction_ballot(pred::Bool)
    lane = (get_sub_group_local_id() - 1) % UInt32
    lo = sub_group_reduce_add(ifelse(pred & (lane < 32), UInt32(1) << lane, UInt32(0)))
    get_max_sub_group_size() <= 32 && return UInt64(lo)
    hi = sub_group_reduce_add(ifelse(pred & (lane >= 32), UInt32(1) << (lane - 32), UInt32(0)))
    return UInt64(lo) | (UInt64(hi) << 32)
end

# Native reductions and scans, for `+` on 32- and 64-bit integers and floats, and `min`/`max`
# on integers (OpenCL's `min` and `max` treat NaN and the sign of zero differently from
# Julia's).
const CollectiveIntTypes = Union{Int32, UInt32, Int64, UInt64}
const CollectiveTypes = Union{CollectiveIntTypes, Float16, Float32, Float64}

@device_override KI.sub_group_reduce(::typeof(+), val::CollectiveTypes) = sub_group_reduce_add(val)
@device_override KI.sub_group_reduce(::typeof(min), val::CollectiveIntTypes) = sub_group_reduce_min(val)
@device_override KI.sub_group_reduce(::typeof(max), val::CollectiveIntTypes) = sub_group_reduce_max(val)

@device_override KI.sub_group_scan(::typeof(+), val::CollectiveTypes) = sub_group_scan_inclusive_add(val)
@device_override KI.sub_group_scan(::typeof(min), val::CollectiveIntTypes) = sub_group_scan_inclusive_min(val)
@device_override KI.sub_group_scan(::typeof(max), val::CollectiveIntTypes) = sub_group_scan_inclusive_max(val)

# the exclusive scans of `cl_khr_subgroups` start from the identity, not from `init`
@device_override function KI.sub_group_exclusive_scan(::typeof(+), val::T, init::T) where {T <: CollectiveTypes}
    prefix = sub_group_scan_exclusive_add(val)
    return ifelse(get_sub_group_local_id() == 1, init, init + prefix)
end

@device_override @inline function KI._print(args...)
    OpenCL._print(args...)
end

end
