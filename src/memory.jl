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

function Base.convert(::Type{CLPtr{T}}, managed::Managed{M}) where {T, M}
    # let null pointers pass through as-is
    ptr = convert(CLPtr{T}, managed.mem)
    if ptr == cl.CL_NULL
        return ptr
    end

    # accessing memory on another queue: ensure the data is ready and take ownership
    if managed.queue != cl.queue()
        maybe_synchronize(managed)
        managed.queue = cl.queue()
    end

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

function alloc(bytes::Int; alignment::Int = 0)
    if cl.memory_backend() == cl.USMBackend()
        return alloc(cl.UnifiedDeviceMemory, bytes; alignment)
    else
        return alloc(cl.SharedVirtualMemory, bytes; alignment)
    end
end

function alloc(::Type{cl.UnifiedDeviceMemory}, bytes::Int; alignment::Int = 0)
    if bytes == 0
        return Managed(cl.UnifiedDeviceMemory(cl.CL_NULL, bytes, cl.context(), cl.device()))
    end
    mem = cl.device_alloc(bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.UnifiedSharedMemory}, bytes::Int; alignment::Int = 0)
    if bytes == 0
        return Managed(cl.UnifiedSharedMemory(cl.CL_NULL, bytes, cl.context(), cl.device()))
    end
    # TODO: support cross-device shared memory (by setting `dev=nothing`)
    mem = cl.shared_alloc(bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.UnifiedHostMemory}, bytes::Int; alignment::Int = 0)
    if bytes == 0
        return Managed(cl.UnifiedHostMemory(cl.CL_NULL, bytes, cl.context()))
    end
    mem = cl.host_alloc(bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.SharedVirtualMemory}, bytes::Int; alignment::Int = 0)
    if bytes == 0
        return Managed(cl.SharedVirtualMemory(cl.CL_NULL, bytes, cl.context()))
    end
    mem = cl.svm_alloc(bytes; alignment)
    return Managed(mem)
end

function free(managed::Managed{<:cl.AbstractMemory})
    mem = managed.mem
    sizeof(mem) == 0 && return

    # XXX: we shouldn't have to look up the device here; all state should be keyed
    #      on the context (i.e., doing `cl.context!` instead of `cl.device!`).
    cl.device!(cl.context(mem).devices[]) do
        # "`clSVMFree` does not wait for previously enqueued commands that may be using
        # svm_pointer to finish before freeing svm_pointer. It is the responsibility of the
        # application to make sure that enqueued commands that use svm_pointer have finished
        # before freeing svm_pointer". USM has `clMemBlockingFreeINTEL`, but by doing the
        # synchronization ourselves we provide more opportunity for concurrent execution.
        synchronize(managed)

        if mem isa cl.SharedVirtualMemory
            cl.svm_free(mem)
        else
            cl.usm_free(mem)
        end
    end

    return
end
