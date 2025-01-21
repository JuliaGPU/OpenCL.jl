# memory operations

## managed memory

# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.

# XXX: immutable with atomic refs?
mutable struct Managed{M}
    const mem::M

    # which stream is currently using the memory.
    queu::cl.CmdQueue

    # whether there are outstanding operations that haven't been synchronized
    dirty::Bool

    # whether the memory has been captured in a way that would make the dirty bit unreliable
    captured::Bool

    function Managed(mem::cl.AbstractMemory; queu = cl.queue(), dirty = true, captured = false)
        # NOTE: memory starts as dirty, because stream-ordered allocations are only
        #       guaranteed to be physically allocated at a synchronization event.
        return new{typeof(mem)}(mem, queu, dirty, captured)
    end
end

Base.sizeof(managed::Managed) = sizeof(managed.mem)

# wait for the current owner of memory to finish processing
function synchronize(managed::Managed)
    cl.finish(managed.queu)
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

function get_backend(x::Managed)
    return cl.get_backend_from_buffer(managed_buftype(x))
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

function Base.unsafe_copyto!(
        ::cl.Context, ::Union{Nothing, cl.Device}, dst::Union{CLPtr{T}, Ptr{T}}, src::Union{CLPtr{T}, Ptr{T}}, N::Integer;
        queu::cl.CmdQueue = cl.queue(), backend = cl.select_backend()
    ) where {T}
    cl.enqueue_abstract_memcpy(dst, src, N * sizeof(T); queu = queu, backend = backend)
    cl.finish(queu)
    return dst
end

function unsafe_fill!(
        ctx::cl.Context, dev::cl.Device, ptr::Union{Ptr{T}, CLPtr{T}},
        pattern::Union{Ptr{T}, CLPtr{T}}, N::Integer; queu::cl.CmdQueue = cl.queue(), backend::Type{<:cl.CLBackend} = cl.select_backend()
    ) where {T}
    pattern_bytes = N * sizeof(T)
    pattern_bytes == 0 && return
    cl.enqueue_abstract_fill(ptr, pattern, sizeof(T), pattern_bytes; queu = queu, backend = backend)
    return cl.finish(queu)
end
