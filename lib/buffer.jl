# OpenCL.Buffer

mutable struct Buffer{T} <: AbstractMemory
    id::cl_mem
    len::Int

    function Buffer{T}(mem_id::cl_mem, len::Integer; retain::Bool=false) where {T}
        if retain
            clRetainMemObject(mem_id)
        end
        buff = new{T}(mem_id, len)
        finalizer(buff) do x
            _finalize(x)
        end
        return buff
    end
end

Base.ndims(b::Buffer) = 1
Base.eltype(b::Buffer{T}) where {T} = T
Base.length(b::Buffer{T}) where {T} = b.len
Base.sizeof(b::Buffer{T}) where {T} = b.len * sizeof(T)


## constructors

# for internal use
function Buffer{T}(len::Int, flags::Integer, hostbuf=nothing;
                   device=:rw, host=:rw) where {T}
    sz = len * sizeof(T)

    if device == :rw
        flags |= CL_MEM_READ_WRITE
    elseif device == :r
        flags |= CL_MEM_READ_ONLY
    elseif device == :w
        flags |= CL_MEM_WRITE_ONLY
    else
        throw(ArgumentError("Device access flag must be one of :rw, :r, or :w"))
    end

    if host == :rw
        # nothing to do
    elseif host == :r
        flags |= CL_MEM_HOST_READ_ONLY
    elseif host == :w
        flags |= CL_MEM_HOST_WRITE_ONLY
    elseif host == :none
        flags |= CL_MEM_HOST_NO_ACCESS
    else
        throw(ArgumentError("Host access flag must be one of :rw, :r, or :w"))
    end

    err_code = Ref{Cint}()
    mem_id = clCreateBuffer(context(), flags, sz, something(hostbuf, C_NULL), err_code)
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end
    return Buffer{T}(mem_id, len)
end

# allocated buffer
function Buffer{T}(len::Integer; host_accessible=false, kwargs...) where {T}
    flags = host_accessible ? CL_MEM_ALLOC_HOST_PTR : 0
    Buffer{T}(len, flags, nothing; kwargs...)
end

# from host memory
function Buffer(hostbuf::Array{T}; copy::Bool=true, kwargs...) where {T}
    flags = copy ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR
    Buffer{T}(length(hostbuf), flags, hostbuf; kwargs...)
end


## memory operations

# reading from buffer to host array, return an event
function Base.unsafe_copyto!(dst::Array{T}, dst_off::Int, src::Buffer{T}, src_off::Int,
                             N::Int; blocking::Bool=false,
                             wait_for::Vector{Event}=Event[]) where T
    nbytes = N * sizeof(T)
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    ret_evt = Ref{cl_event}()
    clEnqueueReadBuffer(queue(), src, blocking, src_off-1, nbytes, pointer(dst, dst_off),
                        n_evts, evt_ids, ret_evt)
    @return_nanny_event(ret_evt[], dst)
end
Base.unsafe_copyto!(dst::Array, src::Buffer, N; kwargs...) =
    unsafe_copyto!(dst, 1, src, 1, N; kwargs...)

# writing from host array to buffer, return an event
function Base.unsafe_copyto!(dst::Buffer{T}, dst_off::Int, src::Array{T}, src_off::Int,
                             N::Int; blocking::Bool=false,
                             wait_for::Vector{Event}=Event[]) where T
    nbytes = N * sizeof(T)
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    ret_evt = Ref{cl_event}()
    clEnqueueWriteBuffer(queue(), dst, blocking, dst_off-1, nbytes, pointer(src, src_off),
                         n_evts, evt_ids, ret_evt)
    @return_nanny_event(ret_evt[], dst)
end
Base.unsafe_copyto!(dst::Buffer, src::Array, N; kwargs...) =
    unsafe_copyto!(dst, 1, src, 1, N; kwargs...)

# copying between two buffers, return an event
function Base.unsafe_copyto!(dst::Buffer{T}, dst_off::Int, src::Buffer{T}, src_off::Int,
                             N::Int; blocking::Bool=false,
                             wait_for::Vector{Event}=Event[]) where T
    nbytes = N * sizeof(T)
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    ret_evt = Ref{cl_event}()
    clEnqueueCopyBuffer(queue(), src, dst, src_off-1, dst_off-1, nbytes,
                        n_evts, evt_ids, ret_evt)
    @return_event ret_evt[]
end
Base.unsafe_copyto!(dst::Buffer, src::Buffer, N; kwargs...) =
    unsafe_copyto!(dst, 1, src, 1, N; kwargs...)

# map a buffer into the host address space and return a (pinned) array and an event
function unsafe_map!(b::Buffer{T}, dims::Dims, flags=:rw; offset::Integer=1,
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
    status  = Ref{Cint}()
    byteoffset = (offset - 1) * sizeof(T)
    mapped  = clEnqueueMapBuffer(queue(), b, blocking,
                                 flags, byteoffset, nbytes,
                                 n_evts, evt_ids, ret_evt, status)
    if status[] != CL_SUCCESS
        throw(CLError(status[]))
    end

    return unsafe_wrap(Array, Ptr{T}(mapped), dims; own=false), Event(ret_evt[])
end

# unmap a buffer, return an event
function unsafe_unmap!(b::Buffer{T}, a::Array{T}; wait_for::Vector{Event}=Event[]) where {T}
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    ret_evt = Ref{cl_event}()
    clEnqueueUnmapMemObject(queue(), b, pointer(a), n_evts, evt_ids, ret_evt)
    return Event(ret_evt[])
end

# fill a buffer with a pattern, returning an event
function unsafe_fill!(b::Buffer{T}, pattern::T, offset::Integer, N::Integer;
                      wait_for::Vector{Event}=Event[]) where {T}
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    ret_evt = Ref{cl_event}()
    nbytes = N * sizeof(T)
    nbytes_pattern = sizeof(T)
    byteoffset = (offset - 1) * sizeof(T)
    @assert nbytes_pattern > 0
    clEnqueueFillBuffer(queue(), b, [pattern],
                        nbytes_pattern, byteoffset, nbytes,
                        n_evts, evt_ids, ret_evt)
    @return_event ret_evt[]
end
unsafe_fill!(b::Buffer, pattern, N::Integer) = unsafe_fill!(b, pattern, 1, N)
