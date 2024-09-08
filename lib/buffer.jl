# OpenCL.Buffer

mutable struct Buffer{T} <: CLMemObject
    valid::Bool
    id::cl_mem
    len::Int
    mapped::Bool
    hostbuf::Ptr{T}

    function Buffer{T}(mem_id::cl_mem, retain::Bool, len::Integer) where T #hostbuf
        @assert len > 0
        @assert mem_id != C_NULL
        if retain
            clRetainMemObject(mem_id)
        end
        nbytes = sizeof(T) * len
        buff = new{T}(true, mem_id, len, false, C_NULL)
        finalizer(buff) do mem_obj
            if !mem_obj.valid
                throw(CLMemoryError("Attempted to double free OpenCL.Buffer $mem_obj"))
            end
            _finalize(mem_obj)
            mem_obj.valid   = false
            mem_obj.mapped  = false
            mem_obj.hostbuf = C_NULL
        end
        return buff
    end
end

Base.ndims(b::Buffer) = 1
Base.eltype(b::Buffer{T}) where {T} = T
Base.length(b::Buffer{T}) where {T} = Int(b.len)
Base.sizeof(b::Buffer{T}) where {T} = Int(b.len * sizeof(T))

Base.show(io::IO, b::Buffer{T}) where {T} = begin
    ptr_val = convert(UInt, Base.pointer(b))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "Buffer{$T}(@$ptr_address)")
end

# XXX: conflict between integer flags and length.
#      design is messy. probably best move all flags into a kwarg?

# high level Buffer constructors with symbol flags
function Buffer(::Type{T}, ctx::Context, len::Integer; hostbuf=nothing) where T
    Buffer(T, ctx, len, (:rw, :null), hostbuf=hostbuf)
end

function Buffer(::Type{T}, ctx::Context, len::Integer, mem_flag::Symbol; hostbuf=nothing) where T
    Buffer(T, ctx, len, (mem_flag, :null), hostbuf=hostbuf)
end

function Buffer(::Type{T}, ctx::Context, len::Integer, mem_flags::NTuple{2, Symbol}; hostbuf=nothing) where T
    f_r  = :r  in mem_flags
    f_w  = :w  in mem_flags
    f_rw = :rw in mem_flags

    if f_r && f_w || f_r && f_rw || f_rw && f_w
        throw(ArgumentError("only one flag in {:r, :w, :rw} can be defined"))
    end

    local flags::cl_mem_flags
    if f_rw && !(f_r || f_w)
        flags = CL_MEM_READ_WRITE
    elseif f_r && !(f_w || f_rw)
        flags = CL_MEM_READ_ONLY
    elseif f_w && !(f_r || f_rw)
        flags = CL_MEM_WRITE_ONLY
    else
        # default buffer is read/write
        flags = CL_MEM_READ_WRITE
    end

    f_alloc = :alloc in mem_flags
    f_use   = :use   in mem_flags
    f_copy  = :copy  in mem_flags
    if f_alloc && f_use || f_alloc && f_copy || f_use && f_copy
        throw(ArgumentError("only one flag in {:alloc, :use, :copy} can be defined"))
    end

    if f_alloc && !(f_use || f_copy)
        flags |= CL_MEM_ALLOC_HOST_PTR
    elseif f_use && !(f_alloc || f_copy)
        flags |= CL_MEM_USE_HOST_PTR
    elseif f_copy && !(f_alloc || f_use)
        flags |= CL_MEM_COPY_HOST_PTR
    end
    return Buffer(T, ctx, len, flags, hostbuf=hostbuf)
end

# low level Buffer constructor with integer parameter flags
function Buffer(::Type{T}, ctx::Context, len::Integer, flags;
                hostbuf::Union{Nothing,Array{T}}=nothing) where T

    if (hostbuf !== nothing &&
        (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)) == 0)
        @warn("'hostbuf' was passed, but no memory flags to make use of it")
    end

    if flags == (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)
        ArgumentError("Use host pointer flag and alloc host pointer flag are mutually exclusive")
    end

    nbytes = 0
    retain_buf::Union{Nothing,Array{T}} = nothing

    if hostbuf !== nothing
        if (flags & CL_MEM_USE_HOST_PTR) != 0
            retain_buf = hostbuf
        end
        if len > length(hostbuf)
            ArgumentError("OpenCL.Buffer specified size greater than host buffer size")
        end
        if len == 0
            len = length(hostbuf)
            nbytes = sizeof(hostbuf)
        else
            nbytes = len * sizeof(T)
        end
    else
        if len <= 0
            ArgumentError("OpenCL.Buffer specified length is <= 0")
        end
        nbytes = len * sizeof(T)
    end

    err_code = Ref{Cint}()
    mem_id = clCreateBuffer(ctx, flags, cl_uint(nbytes),
                            hostbuf !== nothing ? hostbuf : C_NULL,
                            err_code)
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end

    try
        return Buffer{T}(mem_id, false, len)
    catch err
        clReleaseMemObject(mem_id)
        throw(err)
    end
end

# enqueue a read from buffer to hoast array from buffer, return an event
function enqueue_read_buffer(q::CmdQueue,
                             buf::Buffer{T},
                             hostbuf::Array{T},
                             dev_offset::Csize_t,
                             wait_for::Union{Nothing,Vector{Event}},
                             is_blocking::Bool) where T
    n_evts  = wait_for === nothing ? UInt(0) : length(wait_for)
    evt_ids = wait_for === nothing ? C_NULL  : [evt.id for evt in wait_for]
    ret_evt = Ref{cl_event}()
    nbytes  = sizeof(hostbuf)
    @assert nbytes > 0
    clEnqueueReadBuffer(q.id, buf.id, cl_bool(is_blocking),
                                   dev_offset, nbytes, hostbuf,
                                   n_evts, evt_ids, ret_evt)
    @return_nanny_event(ret_evt[], hostbuf)
end

# enqueue a write from host array to buffer, return an event
function enqueue_write_buffer(q::CmdQueue,
                              buf::Buffer{T},
                              hostbuf::Array{T},
                              byte_count::Csize_t,
                              offset::Csize_t,
                              wait_for::Union{Nothing,Vector{Event}},
                              is_blocking::Bool) where T
    n_evts  = wait_for === nothing ? UInt(0) : length(wait_for)
    evt_ids = wait_for === nothing ? C_NULL  : [evt.id for evt in wait_for]
    ret_evt = Ref{cl_event}()
    nbytes  = sizeof(hostbuf)
    @assert nbytes > 0
    clEnqueueWriteBuffer(q.id, buf.id, cl_bool(is_blocking),
                                    offset, nbytes, hostbuf,
                                    n_evts, evt_ids, ret_evt)
    @return_nanny_event(ret_evt[], hostbuf)
end

# enqueue a copy from one buffer to another, return an event
function enqueue_copy_buffer(q::CmdQueue,
                             src::Buffer{T},
                             dst::Buffer{T},
                             byte_count::Csize_t,
                             src_offset::Csize_t,
                             dst_offset::Csize_t,
                             wait_for::Union{Nothing,Vector{Event}}) where T
    n_evts  = wait_for === nothing ? UInt(0) : length(wait_for)
    evt_ids = wait_for === nothing ? C_NULL  : [evt.id for evt in wait_for]
    ret_evt = Ref{cl_event}()
    if byte_count < 0
        byte_count_src = Ref{Csize_t}()
        byte_count_dst = Ref{Csize_t}()
        clGetMemObjectInfo(src.id, CL_MEM_SIZE, sizeof(Csize_t),
                                      byte_count_src, C_NULL)
        clGetMemObjectInfo(src.id, CL_MEM_SIZE, sizeof(Csize_t),
                                      byte_count_dst, C_NULL)
        byte_count = min(byte_count_src[], byte_count_dst[])
    end
    @assert byte_count > 0
    clEnqueueCopyBuffer(q.id, src.id, dst.id,
                                   src_offset, dst_offset, byte_count,
                                   n_evts, evt_ids, ret_evt)
    @return_event ret_evt[]
end

# return whether a given buffer is mapped
ismapped(b::Buffer) = b.mapped

# enqueue an unmap buffer op, return an event
function enqueue_unmap_mem(q::CmdQueue,
                           b::Buffer{T},
                           a::Array{T};
                           wait_for=nothing) where T
    if b.hostbuf != pointer(a)
        throw(ArgumentError("array @$(pointer(a)) is not mapped to buffer $b"))
    end
    if b.mapped == false || b.hostbuf == C_NULL
        throw(CLMemoryError("$b has already been unmapped"))
    end
    n_evts  = 0
    evt_ids = C_NULL
    if wait_for !== nothing
        if isa(wait_for, Event)
            n_evts = 1
            evt_ids = [wait_for.id]
        else
            @assert all([isa(evt, Event) for evt in wait_for])
            n_evts = length(wait_for)
            evt_ids = [evt.id for evt in wait_for]
        end
    end
    ret_evt = Ref{cl_event}()
    clEnqueueUnmapMemObject(q.id, b.id, a,
                                       n_evts, evt_ids, ret_evt)
    b.mapped  = false
    b.hostbuf = C_NULL
    @return_event ret_evt[]
end

# (blocking) unmap a given buffer/array
function unmap!(q::CmdQueue, b::Buffer{T}, a::Array{T}) where T
    evt = enqueue_unmap_mem(q, b, a)
    return wait(evt)
end


# enqueue a memory mapping operation, returning a mapped (pinned) Array and an event
function enqueue_map_mem(q::CmdQueue,
                         b::Buffer{T},
                         flags::Symbol,
                         offset::Integer,
                         dims::Dims,
                         wait_for=nothing,
                         is_blocking=false) where T
    local f::cl_map_flags
    if flags === :r
        f = CL_MAP_READ
    elseif flags === :w
        f = CL_MAP_WRITE
    elseif flags === :rw
        f = CL_MAP_READ | CL_MAP_WRITE
    else
        throw(ArgumentError("enqueue_unmap can have flags of :r, :w, or :rw, got :$flags"))
    end
    return enqueue_map_mem(q, b, f, offset, dims, wait_for, is_blocking)
end

# enqueue a memory mapping operation, returning a mapped (pinned) Array and an event
function enqueue_map_mem(q::CmdQueue,
                         b::Buffer{T},
                         flags::cl_map_flags,
                         offset::Integer,
                         dims::Dims,
                         wait_for=nothing,
                         is_blocking=false) where T
    if length(b) < prod(dims) + offset
        throw(ArgumentError("Buffer length must be greater than or
                             equal to prod(dims) + offset"))
    end
    n_evts  = wait_for === nothing ? cl_uint(0) : cl_uint(length(wait_for))
    evt_ids = wait_for === nothing ? C_NULL  : [evt.id for evt in wait_for]
    flags   = cl_map_flags(flags)
    offset  = unsigned(offset)
    nbytes  = unsigned(prod(dims) * sizeof(T))
    ret_evt = Ref{cl_event}()
    status  = Ref{Cint}()
    mapped  = clEnqueueMapBuffer(q.id, b.id, cl_bool(is_blocking ? 1 : 0),
                                     flags, offset, nbytes,
                                     n_evts, evt_ids, ret_evt, status)
    if status[] != CL_SUCCESS
        throw(CLError(status[]))
    end
    mapped = Base.unsafe_convert(Ptr{T}, mapped)
    N = length(dims)
    local mapped_arr::Array{T, N}
    try
        # julia owns pointer to mapped memory
        mapped_arr = unsafe_wrap(Array{T, N}, mapped, dims, own=false)
        # when array is gc'd, unmap buffer
        b.mapped  = true
        b.hostbuf = mapped
        finalizer(mapped_arr) do x
            if b.mapped && b.hostbuf != C_NULL
                unmap!(q, b, x)
            end
        end
    catch err
        clEnqueueUnmapMemObject(q.id, b.id, mapped,
                                    unsigned(0), C_NULL, C_NULL)
        b.mapped  = false
        b.hostbuf = C_NULL
        rethrow(err)
    end
    return (mapped_arr, Event(ret_evt[]))
end

# low level enqueue fill operation, return event
function enqueue_fill_buffer(q::CmdQueue, buf::Buffer{T},
                             pattern::T, offset::Csize_t,
                             nbytes::Csize_t,
                             wait_for::Union{Vector{Event},Nothing}) where T
    if wait_for === nothing
        evt_ids = C_NULL
        n_evts = cl_uint(0)
    else
        evt_ids = [evt.id for evt in wait_for]
        n_evts  = cl_uint(length(evt_ids))
    end
    ret_evt = Ref{cl_event}()
    nbytes_pattern = sizeof(pattern)
    @assert nbytes_pattern > 0
    clEnqueueFillBuffer(q.id, buf.id, [pattern],
                                   unsigned(nbytes_pattern), offset, nbytes,
                                   n_evts, evt_ids, ret_evt)
    @return_event ret_evt[]
end

# enqueue a fill operation, return an event
function enqueue_fill(q::CmdQueue, buf::Buffer{T}, x::T) where T
    nbytes = sizeof(buf)
    evt = enqueue_fill_buffer(q, buf, x, unsigned(0),
                              unsigned(nbytes), nothing)
    return evt
end

# (blocking) fill the contents of a buffer with with a given value
function fill!(q::CmdQueue, buf::Buffer{T}, x::T) where T
    evt = enqueue_fill(q, buf, x)
    wait(evt)
    return evt
end

# copy the contents of a buffer into an array
function Base.copy!(q::CmdQueue, dst::Array{T}, src::Buffer{T}) where T
    if sizeof(dst) != sizeof(src)
        throw(ArgumentError("Buffer and Array to be copied must be the same size"))
    end
    evt = enqueue_read_buffer(q, src, dst, UInt(0), nothing, true)
    return evt
end

# copy the contents of an array into a buffer
function Base.copy!(q::CmdQueue, dst::Buffer{T}, src::Array{T}) where T
    if sizeof(dst) != sizeof(src)
        throw(ArgumentError("Array and Buffer to be copied must be the same size"))
    end
    nbytes = convert(Csize_t, sizeof(src))
    evt = enqueue_write_buffer(q, dst, src, nbytes, unsigned(0), nothing, true)
    return evt
end

# copy the contents of a buffer into another buffer
function Base.copy!(q::CmdQueue, dst::Buffer{T}, src::Buffer{T}) where T
    if sizeof(dst) != sizeof(src)
        throw(ArgumentError("Buffers to be copied must be the same size"))
    end
    nbytes = convert(Csize_t, sizeof(src))
    evt = enqueue_copy_buffer(q, src, dst, nbytes, unsigned(0),
                              unsigned(0), nothing)
    wait(evt)
    return evt
end

# copy bufer into identical buffer object
function Base.copy(q::CmdQueue, src::Buffer{T}) where T
    nbytes = sizeof(src)
    new_buff = empty_like(q[:context], src)
    copy!(q, new_buff, src)
    return new_buff
end

# create an empty buffer similar to the passed in buffer
function empty_like(ctx::Context, b::Buffer{T}) where T
    len = length(b)
    mf = info(b, :mem_flags)
    if :r in mf
        return Buffer(T, ctx, len, :r)
    elseif :w in mf
        return Buffer(T, ctx, len, :w)
    else
        return Buffer(T, ctx, len, :rw)
    end
end

# create an empty buffer similar to the passed in Array
function empty_like(ctx::Context, a::Array{T}, flag::Symbol=:rw) where T
    len = length(a)
    return Buffer(T, ctx, len, flag)
end

# blocking write of contents of an array to a buffer
function write!(q::CmdQueue, buf::Buffer{T}, hostbuf::Array{T}) where T
    nbytes = unsigned(sizeof(hostbuf))
    enqueue_write_buffer(q, buf, hostbuf, nbytes, unsigned(0), nothing, true)
    return
end

# blocking read of the contents of a buffer into a new array
function read(q::CmdQueue, buf::Buffer{T}) where T
    hostbuf = Vector{T}(undef, length(buf))
    enqueue_read_buffer(q, buf, hostbuf, unsigned(0), nothing, true)
    return hostbuf
end
