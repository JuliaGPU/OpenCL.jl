# high-level memory management

## managed memory

# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.

mutable struct Managed{M}
    const mem::M
    const lock::ReentrantLock

    # which stream is currently using the memory.
    queue::cl.CmdQueue

    # whether there are outstanding operations that haven't been synchronized
    dirty::Bool

    # who is currently using the memory
    user::Symbol

    function Managed(mem::cl.AbstractMemory; queue = cl.queue(), dirty = true, user = :device)
        # NOTE: memory starts as dirty, because stream-ordered allocations are only
        #       guaranteed to be physically allocated at a synchronization event.
        # NOTE: memory also starts as device-owned, because we need to map it as soon as
        #       the host accesses it.
        return new{typeof(mem)}(mem, ReentrantLock(), queue, dirty, user)
    end
end

Base.sizeof(managed::Managed) = sizeof(managed.mem)

function lock_managed(managed::AbstractVector{<:Managed})
    locked = unique(managed)
    sort!(locked; by=memory -> objectid(memory.lock))
    for memory in locked
        lock(memory.lock)
    end
    return locked
end

function unlock_managed(locked::AbstractVector{<:Managed})
    for memory in Iterators.reverse(locked)
        unlock(memory.lock)
    end
    return
end

function with_managed_locks(f::F, managed::AbstractVector{<:Managed}) where {F}
    locked = lock_managed(managed)
    try
        return f()
    finally
        unlock_managed(locked)
    end
end

# wait for the current owner of memory to finish processing
function synchronize(managed::Managed; check_exceptions::Bool=true)
    return Base.@lock managed.lock begin
        cl.finish(managed.queue; check_exceptions)
        managed.dirty = false
        nothing
    end
end

function maybe_synchronize(managed::Managed)
    return Base.@lock managed.lock begin
        if managed.dirty
            synchronize(managed)
        end
        nothing
    end
end

# transfer queue ownership of an allocation, synchronizing the previous owner if needed,
# and mark it dirty in anticipation of a device-side operation. the caller must hold
# `managed.lock`, and keep holding it until that operation has been submitted to `queue`:
# the recorded ownership describes the submitted operation, so another task may only
# observe it together with the submission it describes.
function take_ownership!(managed::Managed{M}; queue=cl.queue()) where {M}
    sizeof(managed) == 0 && return managed

    # accessing memory on another queue: ensure the data is ready and take ownership
    if managed.queue != queue
        managed.dirty && synchronize(managed)
        managed.queue = queue
    end

    # coarse-grained SVM needs to be unmapped when accessing it back from the device
    # TODO: support fine-grained SVM
    if M == cl.SharedVirtualMemory && managed.user == :host
        cl.enqueue_svm_unmap(pointer(managed.mem); queue)
        managed.user = :device
    end

    managed.dirty = true
    return managed
end

function device_convert(typ::Union{Type{<:CLPtr}, Type{cl.Buffer}},
                        managed::Managed{M}) where {M}
    # let null pointers pass through as-is
    # XXX: does not work for buffers
    ptr = convert(typ, managed.mem)
    if ptr == cl.CL_NULL
        return ptr
    end

    take_ownership!(managed)
    return ptr
end

Base.convert(typ::Union{Type{<:CLPtr}, Type{cl.Buffer}}, managed::Managed) =
    Base.@lock managed.lock device_convert(typ, managed)

function Base.convert(typ::Type{<:Ptr}, managed::Managed{M}) where {M}
    # let null pointers pass through as-is
    ptr = convert(typ, managed.mem)
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

    return Base.@lock managed.lock begin
        # make sure any work on the memory has finished.
        managed.dirty && synchronize(managed)

        # coarse-grained SVM needs to be mapped when initially accessing it from the host
        # TODO: support fine-grained SVM
        if M == cl.SharedVirtualMemory && managed.user != :host
            cl.enqueue_svm_map(pointer(managed.mem), sizeof(managed.mem), :rw; blocking=true)
            managed.user = :host
        end

        ptr
    end
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
function managed_alloc(t::Type{T}, bytes::Int; kwargs...) where T
    if bytes == 0
        return Managed(T())
    else
        alloc(t, bytes; kwargs...)
    end
end

function alloc(::Type{cl.UnifiedDeviceMemory}, bytes::Int; alignment::Int = 0)
    mem = cl.device_alloc(bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.UnifiedSharedMemory}, bytes::Int; alignment::Int = 0)
    # TODO: support cross-device shared memory (by setting `dev=nothing`)
    mem = cl.shared_alloc(bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.UnifiedHostMemory}, bytes::Int; alignment::Int = 0)
    mem = cl.host_alloc(bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.SharedVirtualMemory}, bytes::Int; alignment::Int = 0)
    mem = cl.svm_alloc(bytes; alignment)
    return Managed(mem)
end

function alloc(::Type{cl.Buffer}, bytes::Int; alignment::Int = 0)
    # TODO: use alignment
    buf = cl.Buffer(bytes)
    return Managed(buf)
end

function free(managed::Managed)
    sizeof(managed) == 0 && return
    return Base.@lock managed.lock begin
        mem = managed.mem
        # "`clSVMFree` does not wait for previously enqueued commands that may be using
        # svm_pointer to finish before freeing svm_pointer. It is the responsibility of the
        # application to make sure that enqueued commands that use svm_pointer have finished
        # before freeing svm_pointer". USM has `clMemBlockingFreeINTEL`, but by doing the
        # synchronization ourselves we provide more opportunity for concurrent execution.
        if managed.queue.valid
            # this may run from a finalizer, where a device-side exception cannot be thrown
            synchronize(managed; check_exceptions=false)
        end

        if mem isa cl.SharedVirtualMemory
            if managed.user == :host && managed.queue.valid
                # Finalizers must not query or mutate task-local state, so use the queue owned by
                # the allocation. Finish the unmap before releasing the SVM allocation.
                cl.enqueue_svm_unmap(pointer(mem); queue=managed.queue)
                cl.finish(managed.queue; check_exceptions=false)
            end
            cl.svm_free(mem)
        elseif mem isa cl.UnifiedMemory
            cl.usm_free(mem)
        else
            cl.release(mem)
        end

        nothing
    end
end
