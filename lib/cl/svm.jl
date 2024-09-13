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
function enqueue_svm_memcpy(dst::Ptr, src::Ptr, nbytes::Integer; blocking::Bool=false,
                             wait_for::Vector{Event}=Event[])
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMemcpy(queue(), blocking, dst, src, nbytes, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

# map an SVM buffer into the host address space, returning an event
function enqueue_svm_map(ptr::Ptr, nbytes::Integer, flags=:rw; blocking::Bool=false,
                          wait_for::Vector{Event}=Event[])
    flags = if flags == :rw
        CL_MAP_READ | CL_MAP_WRITE
    elseif flags == :r
        CL_MAP_READ
    elseif flags == :w
        CL_MAP_WRITE
    else
        throw(ArgumentError("enqueue_unmap can have flags of :r, :w, or :rw, got :$flags"))
    end
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMap(queue(), blocking, flags, ptr, nbytes,
                        n_evts, evt_ids, ret_evt)

        return Event(ret_evt[])
    end
end

# unmap a buffer, returning an event
function enqueue_svm_unmap(ptr::Ptr; wait_for::Vector{Event}=Event[])
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMUnmap(queue(), ptr, n_evts, evt_ids, ret_evt)
        return Event(ret_evt[])
    end
end

# fill a buffer with a pattern, returning an event
function enqueue_svm_fill(ptr::Ptr, pattern::T, N::Integer;
                           wait_for::Vector{Event}=Event[]) where {T}
    nbytes = N * sizeof(T)
    nbytes_pattern = sizeof(T)
    @assert nbytes_pattern > 0
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMemFill(queue(), ptr, [pattern],
                            nbytes_pattern, nbytes,
                            n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end
