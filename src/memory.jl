# high-level memory management

## managed memory

# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.

# XXX: immutable with atomic refs?
mutable struct Managed{M}
    const mem::M

    # which stream is currently using the memory.
    queue::cl.CmdQueue

    # whether there are outstanding operations that haven't been synchronized
    dirty::Bool

    # whether the memory has been captured in a way that would make the dirty bit unreliable
    captured::Bool

    function Managed(mem::cl.AbstractMemory; queue = cl.queue(), dirty = true, captured = false)
        # NOTE: memory starts as dirty, because stream-ordered allocations are only
        #       guaranteed to be physically allocated at a synchronization event.
        return new{typeof(mem)}(mem, queue, dirty, captured)
    end
end

Base.sizeof(managed::Managed) = sizeof(managed.mem)

# wait for the current owner of memory to finish processing
function synchronize(managed::Managed)
    cl.finish(managed.queue)
    return managed.dirty = false
end

function maybe_synchronize(managed::Managed)
    return if managed.dirty || managed.captured
        synchronize(managed)
    end
end

function managed_buftype(::Managed{M}) where {M}
    return M
end

function Base.convert(::Type{CLPtr{T}}, managed::Managed{M}) where {T, M}
    # let null pointers pass through as-is
    ptr = convert(CLPtr{T}, managed.mem)
    if ptr == cl.CL_NULL
        return ptr
    end

    # TODO: FIGURE OUT ACTIVE STATE

    managed.dirty = true
    return ptr
end

function Base.convert(::Type{Ptr{T}}, managed::Managed{M}) where {T, M}
    # let null pointers pass through as-is
    ptr = convert(Ptr{T}, managed.mem)
    if ptr == C_NULL
        return ptr
    end

    # accessing memory on the CPU: only allowed for host or unified allocations
    if M == cl.UnifiedDeviceMemory
        throw(
            ArgumentError(
                """cannot take the CPU address of GPU memory."""
            )
        )

    end

    # make sure any work on the memory has finished.
    maybe_synchronize(managed)
    return ptr
end


## OOM handling

export OutOfGPUMemoryError

"""
    OutOfGPUMemoryError()

An operation allocated too much GPU memory.
"""
struct OutOfGPUMemoryError <: Exception
    sz::Int
    dev::cl.Device

    function OutOfGPUMemoryError(sz::Integer = 0, dev::cl.Device = cl.device())
        return new(sz, dev)
    end
end

function Base.showerror(io::IO, err::OutOfGPUMemoryError)
    print(io, "Out of GPU memory")
    if err.sz > 0
        print(io, " trying to allocate $(Base.format_bytes(err.sz))")
    end
    print(" on device $((err.dev).name)")
    #=
    if length(memory_properties(err.dev)) == 1
        # XXX: how to handle multiple memories?
        print(" with $(Base.format_bytes(only(memory_properties(err.dev)).totalSize))")
    end
    =#
    return io
end


## public interface

function alloc(ctx, dev, bytes::Int; alignment::Int = 0)
    if cl.device_state(dev).usm
        return cl.alloc(cl.UnifiedDeviceMemory, ctx, dev, bytes; alignment)
    else
        return cl.alloc(cl.SharedVirtualMemory, ctx, dev, bytes; alignment)
    end
end

function alloc(::Type{cl.UnifiedDeviceMemory}, ctx, dev, bytes::Int; alignment::Int = 0)
    bytes == 0 && return Managed(cl.UnifiedDeviceMemory(cl.CL_NULL, bytes, ctx, dev))

    mem = cl.device_alloc(ctx, dev, bytes; alignment)
    # make_resident(ctx, dev, mem)
    return Managed(mem)
end

function alloc(::Type{cl.UnifiedSharedMemory}, ctx, dev, bytes::Int; alignment::Int = 0)
    bytes == 0 && return Managed(cl.UnifiedSharedMemory(cl.CL_NULL, bytes, ctx, dev))

    # TODO: support cross-device shared memory (by setting `dev=nothing`)

    mem = cl.shared_alloc(ctx, dev, bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.UnifiedHostMemory}, ctx, dev, bytes::Int; alignment::Int = 0)
    bytes == 0 && return Managed(cl.UnifiedHostMemory(cl.CL_NULL, bytes, ctx))

    mem = cl.host_alloc(ctx, bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.SharedVirtualMemory}, ctx, dev, bytes::Int; alignment::Int = 0)
    bytes == 0 && return Managed(cl.SharedVirtualMemory(cl.CL_NULL, bytes, ctx))

    mem = cl.svm_alloc(ctx, bytes; alignment)
    # make_resident(ctx, dev, mem)
    return Managed(mem)
end

function free(managed::Managed{<:cl.AbstractMemory})
    mem = managed.mem

    sizeof(mem) == 0 && return

    # XXX: is it necessary to evice memory if we are going to free it?
    #      this is racy, because eviction is not queue-ordered, and
    #      we don't want to synchronize inside what could have been a
    #      GC-driven finalizer. if we need to, port the stream/queue
    #      tracking from CUDA.jl so that we can synchronize only the
    #      queue that's associated with the memory.
    #if mem isa oneL0.UnifiedDeviceMemory || mem isa oneL0.UnifiedSharedMemory
    #    ctx = oneL0.context(mem)
    #    dev = oneL0.device(mem)
    #    evict(ctx, dev, mem)
    #end

    if mem isa cl.SharedVirtualMemory
        cl.svm_free(mem)
        finish(queue())
    else
        cl.usm_free(mem; blocking = true)
    end

    # TODO: queue-ordered free from non-finalizer tasks once we have
    #       `zeMemFreeAsync(ptr, queue)`

    return
end
