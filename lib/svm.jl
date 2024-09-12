mutable struct SVMBuffer{T}
    ptr::Ptr{T}
    len::Int

    function SVMBuffer{T}(len::Integer, access::Symbol=:rw;
                          fine_grained=false, alignment=nothing) where {T}
        flags = if access == :rw
            CL_MEM_READ_WRITE
        elseif access == :r
            CL_MEM_READ_ONLY
        elseif access == :w
            CL_MEM_WRITE_ONLY
        else
            throw(ArgumentError("Invalid access type"))
        end

        if fine_grained
            flags |= CL_MEM_SVM_FINE_GRAIN_BUFFER
        end

        ptr = clSVMAlloc(context(), flags, len * sizeof(T), something(alignment, 0))
        obj = new{T}(ptr, len)
        finalizer(obj) do x
            # TODO: asynchronous free using clEnqueueSVMFree?
            clSVMFree(context(), x)
        end

        return obj
    end
end

Base.unsafe_convert(::Type{Ptr{T}}, x::SVMBuffer) where {T} = convert(Ptr{T}, x.ptr)
@inline function Base.pointer(x::SVMBuffer{T}, i::Integer=1) where T
    Base.unsafe_convert(Ptr{T}, x) + (i-1)*sizeof(T)
end

Base.ndims(b::SVMBuffer) = 1
Base.eltype(b::SVMBuffer{T}) where {T} = T
Base.length(b::SVMBuffer{T}) where {T} = b.len
Base.sizeof(b::SVMBuffer{T}) where {T} = b.len * sizeof(T)


## memory operations

# these generally only make sense for coarse-grained SVM buffers;
# fine-grained buffers can just be used directly.

# copy from and to SVM buffers
for (srcty, dstty) in [(:Array, :SVMBuffer), (:SVMBuffer, :Array), (:SVMBuffer, :SVMBuffer)]
    @eval begin
        function Base.unsafe_copyto!(dst::$dstty{T}, dst_off::Int, src::$srcty{T}, src_off::Int,
                                     N::Int; blocking::Bool=false,
                                     wait_for::Vector{Event}=Event[]) where T
            nbytes = N * sizeof(T)
            n_evts  = length(wait_for)
            evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
            ret_evt = Ref{cl_event}()
            clEnqueueSVMMemcpy(queue(), blocking, pointer(dst, dst_off),
                               pointer(src, src_off), nbytes, n_evts, evt_ids, ret_evt)
            @return_nanny_event(ret_evt[], dst)
        end
        Base.unsafe_copyto!(dst::$dstty, src::$srcty, N; kwargs...) =
            unsafe_copyto!(dst, 1, src, 1, N; kwargs...)
    end
end

# map an SVM buffer into the host address space and return a (pinned) array and an event
function unsafe_map!(b::SVMBuffer{T}, dims::Dims, flags=:rw; offset::Integer=1,
                     blocking::Bool=false, wait_for::Vector{Event}=Event[]) where {T}
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    flags = if flags == :rw
        CL_MAP_READ | CL_MAP_WRITE
    elseif flags == :r
        CL_MAP_READ
    elseif flags == :w
        CL_MAP_WRITE
    else
        throw(ArgumentError("enqueue_unmap can have flags of :r, :w, or :rw, got :$flags"))
    end
    nbytes  = prod(dims) * sizeof(T)
    ret_evt = Ref{cl_event}()
    clEnqueueSVMMap(queue(), blocking, flags, pointer(b, offset), nbytes,
                    n_evts, evt_ids, ret_evt)

    return unsafe_wrap(Array, pointer(b, offset), dims; own=false), Event(ret_evt[])
end

# unmap a buffer, return an event
function unsafe_unmap!(b::SVMBuffer{T}, a::Array{T}; wait_for::Vector{Event}=Event[]) where {T}
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    ret_evt = Ref{cl_event}()
    clEnqueueSVMUnmap(queue(), pointer(a), n_evts, evt_ids, ret_evt)
    return Event(ret_evt[])
end

# fill a buffer with a pattern, returning an event
function unsafe_fill!(b::SVMBuffer{T}, pattern::T, offset::Integer, N::Integer;
                      wait_for::Vector{Event}=Event[]) where {T}
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    ret_evt = Ref{cl_event}()
    nbytes = N * sizeof(T)
    nbytes_pattern = sizeof(T)
    @assert nbytes_pattern > 0
    clEnqueueSVMMemFill(queue(), pointer(b, offset), [pattern],
                        nbytes_pattern, nbytes,
                        n_evts, evt_ids, ret_evt)
    @return_event ret_evt[]
end
unsafe_fill!(b::SVMBuffer, pattern, N::Integer) = unsafe_fill!(b, pattern, 1, N)
