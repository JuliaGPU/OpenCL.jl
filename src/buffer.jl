
type Buffer{T} <: CLMemObject
    valid::Bool
    id::CL_mem
    size::CL_uint
    hostbuf

    function Buffer(mem_id::CL_mem; retain=true, hostbuf=nothing)
        if retain
            @check api.clRetainMemObject(mem_id)
        end
        buff = new(true, mem_id, hostbuf)
        finalizer(buff, mem_obj -> if mem_obj.valid release!(mem_obj) end)
        return buff
    end
end

Base.length{T}(b::Buffer{T}) = (b.size / sizeof(T))
Base.ndims(b::Buffer) = 1
Base.eltype{T}(b::Buffer{T}) = T
Base.isnull{T}(b::Buffer) = (b.id == C_NULL)


function create_cl_buffer(ctx::CL_context, flags::CL_mem_flags,
                          size::Csize_t, host_buffer::Ptr{Void})
    status = Array(C_int, 1)
    mem_id = api.clCreateBuffer(ctx, flags, size, host_ptr, status)
    if status[1] != CL_SUCCESS
        throw(CLError(status[1]))
    end
    return mem_id
end

function Buffer(ctx::Context, flags::CL_mem_flags, size::Csize_t=0; hostbuf=nothing)
    if (hostbuf != nothing && 
        !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)))
        warn("'hostbuf' was passed, but no memory flags to make use of it")
    end
    
    buf_ptr::Ptr{Void} = C_NULL
    retain_buf = nothing

    if hostbuf != nothing
        buf_ptr = convert(Ptr{Void}, hostbuf)
        if flags & CL_MEM_USE_HOST_PTR
            retain_buf = hostbuf
        end
        if size > sizeof(hostbuf)
            error("OpenCL.Buffer specified size greater than host buffer size")
        end
        if size == 0
            size = sizeof(hostbuf)
        end
    end
    
    mem_id = create_cl_buffer(ctx.id, flags, size, buf_ptr)
    
    try
        return Buffer(mem_id, 
                      retain=false,
                      hostbuf=retain_buf)
    catch err
        @check api.clReleaseMemObject(mem_id)
        throw(err)
    end
end

function copy!{T}(dst::Array{T}, src::Buffer{T})
    if length(dist) != length(src)
        throw(ArgumentError("Inconsistent array length"))
    end
    nbytes = length(src) * sizeof(T)
    # copy
    return dst
end

function copy!{T}(dst::Buffer{T}, src::Array{T})
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length"))
    end
    nbytes = length(src) * sizeof(T)
    # copy
    return dst
end

function enqueue_read_buffer(q::Queue, 
                             buf::Buffer, 
                             hostbuf::Array,
                             dev_offset::Csize_t,
                             wait_for::Vector{Event},
                             is_blocking::Bool)

    evt_ids = [evt.id for evt in wait_for]
    n_evts  = cl_uint(length(evt_ids))
    ret_evt = Array(CL_event, 1)
    nbytes  = unsigned(sizeof(hostbuf))
    @check api.clEnqueueReadBuffer(ctx.id, buf.id, cl_bool(is_blocking),
                                   dev_offset, nbytes, buf_ptr,
                                   n_evts, n_evts == 0 ? C_NULL : evt_ids, 
                                   ret_evt)
    #TODO: nanny event
    @return_event ret_evt[1] 
end

function enqueue_write_buffer(q::Queue,
                              buf::Buffer,
                              hostbuf::Array,
                              byte_count::Csize_t,
                              src_offset::Csize_t,
                              dst_offset::Csize_t,
                              wait_for::Vector{Event})

    evt_ids = [evt.id for evt in wait_for]
    n_evts  = cl_uint(length(evt_ids))
    ret_evt = Array(CL_event, 1)
    nbytes  = unsigned(sizeof(hostbuf))
    @check api.clEnqueueWriteBuffer(ctx.id, buf.id, cl_bool(is_blocking),
                                    dev_offset, nbytes, hostbuf,
                                    n_evts, n_evts == 0 ? C_NULL : evt_ids,
                                    ret_evt)
    # TODO: nanny evt
    @return_event ret_evt[1]
end

function enqueue_copy_buffer(q::Queue,
                             src::Buffer,
                             dst::Buffer,
                             byte_count::Csize_t,
                             src_offset::Csize_t,
                             dst_offset::Csize_t,
                             wait_for::Vector{Event})

    evt_ids = [evt.id for evt in wait_for]
    n_evts  = cl_uint(length(evt_ids))
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
    @check api.clEnqueueCopyBuffer(ctx.id, src.id, dst.id,
                                   src_offset, dst_offset, byte_count,
                                   n_evts, n_evts == 0 ? C_NULL : evt_ids, ret_evt)
    @return_event ret_evt[1] 
end

function enqueue_map_buffer(q::Queue, b::Buffer, flags, offset, shape,
                            wait_for=nothing, is_blocking=false)
#TODO:
end

@ocl_v1_2_only begin
    function enqueue_fill_buffer(q::Queue, buf::Buffer, pattern::Array,
                                 offset::Csize_t, nbytes::Csize_t,
                                 wait_for::Vector{Event})

        evt_ids = [evt.id for evt in wait_for]
        n_evts  = cl_uint(length(evt_ids))
        ret_evt = Array(CL_event, 1)
        nbytes_pattern  = unsigned(sizeof(pattern)) 
        @check api.clEnqueueFillBuffer(ctx.id, buf.id,
                                       pattern, nbytes_pattern,
                                       offset, size,
                                       n_evts, n_evts == 0 ? C_NULL : evt_ids,
                                       ret_evt)
        # TODO: nanny evt
        @return_event ret_evt[1]
    end
end


#function write!{T}(q::Queue, b::Buffer, v::Vector{T})
#    csize = convert(Csize_t, b.size)
#    vptr = convert(Ptr{Void}, v)
#    clEnqueueWriteBuffer(q.id, b.ptr, CL_TRUE, 0, csize, vptr, 0, C_NULL, C_NULL)
#end 

#function read{T}(q::Queue, b::Buffer{T})
#    vptr = convert(Ptr{Void}, Array(T, b.size / sizeof(T)))
#    csize = convert(Csize_t, b.size)
#    clEnqueueReadBuffer(q.id, b.ptr, CL_TRUE, 0, csize, vptr, 0, C_NULL, C_NULL) 
#    return unsafe_ref(convert(Ptr{T}, vptr))
#end
