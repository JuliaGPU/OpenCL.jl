
type Buffer{T} <: CLMemObject
    valid::Bool
    ptr::CL_mem
    size::CL_uint
    hostbuf

    function Buffer(mem_ptr::CL_mem; retain=true, hostbuf=nothing)
        if retain
            @check api.clRetainMemObject(mem_ptr)
        end
        buff = new(true, mem_ptr, hostbuf)
        finalizer(buff, mem_obj -> if mem_obj.valid release!(mem_obj) end)
        return buff
    end
end

Base.length{T}(b::Buffer{T}) = (b.size / sizeof(T))
Base.ndims(b::Buffer) = 1
Base.eltype{T}(b::Buffer{T}) = T
Base.isnull{T}(b::Buffer) = (b.ptr == C_NULL)


function create_cl_buffer(ctx::CL_context, flags::CL_mem_flags,
                          size::Csize_t, host_buffer::Ptr{Void})
    status = Array(C_int, 1)
    mem_ptr = api.clCreateBuffer(ctx, flags, size, host_ptr, status)
    if status[1] != CL_SUCCESS
        throw(CLError(status[1]))
    end
    return mem_ptr
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
    
    mem_ptr = create_cl_buffer(ctx.id, flags, size, buf_ptr)
    
    try
        return Buffer(mem_ptr, 
                      retain=false,
                      hostbuf=retain_buf)
    catch err
        @check api.clReleaseMemObject(mem_ptr)
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

function enqueue_fill_buffer(q::Queue, mem, pattern, offset, size, wait_for=nothing)
end

function enqueue_read_buffer(q::Queue, mem, hostbuf::Buffer)
end

function enqueue_write_buffer(q::Queue, mem, hostbuf::Buffer)
end

function enqueue_copy_buffer(q::Queue, mem, hostbuf::Buffer)
end

function enqueue_map_buffer(q::Queue, b::Buffer, flags, offset, shape,
                            wait_for=nothing, is_blocking=false)
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
