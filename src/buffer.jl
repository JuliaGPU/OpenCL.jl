# --- OpenCL Buffer ---

type Buffer{T} <: CLMemObject
    valid::Bool
    id::CL_mem
    size::CL_uint
    hostbuf::Union(Nothing, Array{T})

    function Buffer(mem_id::CL_mem, retain::Bool, size::CL_uint) #hostbuf
        if retain
            @check api.clRetainMemObject(mem_id)
        end
        buff = new(true, mem_id, size, nothing)
        finalizer(buff, mem_obj -> begin 
            if !mem_obj.valid
                error("attempted to double free $mem_obj")
            end
            release!(mem_obj)
            mem_obj.valid   = false
            mem_obj.hostbuf = nothing
        end)
        return buff
    end
end

Base.length{T}(b::Buffer{T}) = int(b.size / sizeof(T))
Base.ndims(b::Buffer) = 1
Base.eltype{T}(b::Buffer{T}) = T

function Buffer{T}(::Type{T}, ctx::Context, nbytes=0; hostbuf=nothing)
    Buffer(T, ctx, (:rw, :null), nbytes, hostbuf=hostbuf)
end

function Buffer{T}(::Type{T}, ctx::Context, mem_flag::Symbol, nbytes=0; hostbuf=nothing)
    Buffer(T, ctx, (mem_flag, :null), nbytes, hostbuf=hostbuf)
end

function Buffer{T}(::Type{T}, ctx::Context, mem_flags::NTuple{2, Symbol}, nbytes=0; hostbuf=nothing)
    f_r  = :r  in mem_flags
    f_w  = :w  in mem_flags
    f_rw = :rw in mem_flags

    if f_r && f_w || f_r && f_rw || f_rw && f_w
        throw(ArgumentError("only one flag in {:r, :w, :rw} can be defined"))
    end

    flags::CL_mem_flags
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
    return Buffer(T, ctx, flags, nbytes, hostbuf=hostbuf)
end

function Buffer{T}(::Type{T}, ctx::Context, flags::CL_mem_flags, nbytes=0;
                   hostbuf::Union(Nothing, Array{T})=nothing)
    if (hostbuf != nothing && 
        !bool((flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))))
        warn("'hostbuf' was passed, but no memory flags to make use of it")
    end
    
    retain_buf::Union(Nothing, Array{T}) = nothing
    if hostbuf != nothing
        if bool(flags & CL_MEM_USE_HOST_PTR)
            retain_buf = hostbuf
        end
        if nbytes > sizeof(hostbuf)
            error("OpenCL.Buffer specified size greater than host buffer size")
        end
        if nbytes == 0
            nbytes = sizeof(hostbuf)
        end
    end

    if nbytes <= 0
        error("OpenCL.Buffer specified size is <= 0 bytes")
    end
    nbytes = cl_uint(nbytes)

    err_code = Array(CL_int, 1)
    mem_id = api.clCreateBuffer(ctx.id, flags, nbytes,
                                hostbuf != nothing ? hostbuf : C_NULL, 
                                err_code)
    if err_code[1] != CL_SUCCESS
        throw(CLError(err_code[1]))
    end

    try
        return Buffer{T}(mem_id, false, nbytes)
    catch err
        api.clReleaseMemObject(mem_id)
        throw(err)
    end
end

function enqueue_read_buffer{T}(q::CmdQueue, 
                                buf::Buffer{T}, 
                                hostbuf::Array{T},
                                dev_offset::Csize_t,
                                wait_for::Union(Nothing, Vector{Event}),
                                is_blocking::Bool)
    n_evts  = wait_for == nothing ? uint(0) : length(wait_for) 
    evt_ids = wait_for == nothing ? C_NULL  : [evt.id for evt in wait_for]
    ret_evt = Array(CL_event, 1)
    nbytes  = sizeof(hostbuf)
    @check api.clEnqueueReadBuffer(q.id, buf.id, cl_bool(is_blocking),
                                   dev_offset, nbytes, hostbuf,
                                   n_evts, evt_ids, ret_evt)
    @return_nanny_event(ret_evt[1], hostbuf) 
end

function enqueue_write_buffer{T}(q::CmdQueue,
                                 buf::Buffer{T},
                                 hostbuf::Array{T},
                                 byte_count::Csize_t,
                                 offset::Csize_t,
                                 wait_for::Union(Nothing, Vector{Event}),
                                 is_blocking::Bool)
    n_evts  = wait_for == nothing ? uint(0) : length(wait_for) 
    evt_ids = wait_for == nothing ? C_NULL  : [evt.id for evt in wait_for]
    ret_evt = Array(CL_event, 1)
    nbytes  = unsigned(sizeof(hostbuf))
    @check api.clEnqueueWriteBuffer(q.id, buf.id, cl_bool(is_blocking),
                                    offset, nbytes, hostbuf,
                                    n_evts, evt_ids, ret_evt)
    buf.size = nbytes
    @return_nanny_event(ret_evt[1], hostbuf)
end

function enqueue_copy_buffer{T}(q::CmdQueue,
                                src::Buffer{T},
                                dst::Buffer{T},
                                byte_count::Csize_t,
                                src_offset::Csize_t,
                                dst_offset::Csize_t,
                                wait_for::Union(Nothing, Vector{Event}))
    n_evts  = wait_for == nothing ? uint(0) : length(wait_for) 
    evt_ids = wait_for == nothing ? C_NULL  : [evt.id for evt in wait_for]
    ret_evt = Array(CL_event, 1)
    if byte_count < 0
        byte_count_src = Array(Csize_t, 1)
        byte_count_dst = Array(Csize_t, 1)
        @check api.clGetMemObjectInfo(src.id, CL_MEM_SIZE, sizeof(Csize_t),
                                      byte_count_src, C_NULL)
        @check api.clGetMemObjectInfo(src.id, CL_MEM_SIZE, sizeof(Csize_t),
                                      byte_count_dst, C_NULL)
        byte_count = min(byte_count_src[1], byte_count_dst[1])
    end
    @check api.clEnqueueCopyBuffer(q.id, src.id, dst.id,
                                   src_offset, dst_offset, byte_count,
                                   n_evts, evt_ids, ret_evt)
    @return_event ret_evt[1] 
end

function enqueue_map_buffer(q::CmdQueue, b::Buffer, flags, offset, shape,
                            wait_for=nothing, is_blocking=false)
#TODO:
end

@ocl_v1_2_only begin
    function enqueue_fill_buffer{T}(q::CmdQueue, buf::Buffer{T}, pattern::T,
                                    offset::Csize_t, nbytes::Csize_t,
                                    wait_for::Union(Vector{Event}, Nothing))
        
        if wait_for == nothing
            evt_ids = C_NULL
            n_evts = cl_uint(0)
        else
            evt_ids = [evt.id for evt in wait_for]
            n_evts  = cl_uint(length(evt_ids))
        end
        ret_evt = Array(CL_event, 1)
        nbytes_pattern  = unsigned(sizeof(pattern)) 
        @check api.clEnqueueFillBuffer(q.id, buf.id, [pattern], 
                                       nbytes_pattern, offset, buf.size,
                                       n_evts, evt_ids, ret_evt)
        # TODO: nanny evt
        @return_event ret_evt[1]
    end

    function enqueue_fill{T}(q::CmdQueue, buf::Buffer{T}, x::T)
        nbytes = uint(buf.size)
        evt = enqueue_fill_buffer(q, buf, x, unsigned(0), nbytes, nothing)
        return evt
    end
    
    function fill!{T}(q::CmdQueue, buf::Buffer{T}, x::T)
        wait(enqueue_fill(q, buf, x))
    end
end

function copy!{T}(q::CmdQueue, dst::Array{T}, src::Buffer{T})
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length"))
    end
    #TODO: change to sizeof
    nbytes = length(src) * sizeof(T)
    evt = enqueue_read_buffer(q, src, dst, uint(0), nothing, true)
    return evt
end

function copy!{T}(q::CmdQueue, dst::Buffer{T}, src::Array{T})
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length"))
    end
    nbytes = convert(Csize_t, sizeof(src))
    evt = enqueue_write_buffer(q, dst, src, nbytes, unsigned(0), nothing, true)
    return evt
end
#function enqueue_write_buffer{T}(q::CmdQueue,
#                                 buf::Buffer{T},
#                                 hostbuf::Array{T},
#                                 byte_count::Csize_t,
#                                 offset::Csize_t,
#                                 wait_for::Union(Nothing, Vector{Event}),
#                                 is_blocking::Bool)
 
function copy!{T}(q::CmdQueue, dst::Buffer{T}, src::Buffer{T})
    if length(dst) != length(src)
        throw(ArgumentError("Buffers to be copied must be the same length"))
    end
    #TODO: change to sizeof
    nbytes = length(src) * sizeof(T)
    evt = enqueue_copy_buffer(q, src, dst, sizeof(src), unsigned(0), unsigned(0), nothing, true)
    return evt
end

# copy bufer into identical buffer object
function copy{T}(q::CmdQueue, src::Buffer{T})
   #TODO:
end

function empty{T}(::Type{T}, ctx::Context, dims)
    size = sizeof(T)
    for d in dims
        if d < 1
            throw(ArgumentError("all dims must be greater than or equal to 1"))
        end
        size *= d
    end
    if size <= 0
        error("OpenCL.Buffer specified size is <= 0 bytes")
    end
    size = cl_uint(size)

    err_code = Array(CL_int, 1)
    mem_id = api.clCreateBuffer(ctx.id, CL_MEM_READ_WRITE, size, C_NULL, err_code)
    if err_code[1] != CL_SUCCESS
        throw(CLError(err_code[1]))
    end

    try
        return Buffer{T}(mem_id, false, size)
    catch err
        api.clReleaseMemObject(mem_id)
        throw(err)
    end
end

#TODO: enqueue low level functions should match up signature with cl.api
function write!{T}(q::CmdQueue, buf::Buffer{T}, hostbuf::Array{T})
    nbytes = unsigned(sizeof(hostbuf))
    @assert nbytes == buf.size
    evt = enqueue_write_buffer(q, buf, hostbuf, nbytes, unsigned(0), nothing, true)
    wait(evt)
end 

function read{T}(q::CmdQueue, buf::Buffer{T})
    hostbuf = Array(T, int(buf.size / sizeof(T)))
    enqueue_read_buffer(q, buf, hostbuf, unsigned(0), nothing, true)
    return hostbuf
end
