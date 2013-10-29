# --- low level buffer implementation ----

immutable Buffer{T} <: CLMemObject
    ptr::CL_mem
    size::CL_uint
end

Base.length{T}(b::Buffer{T}) = (b.size / sizeof(T))
Base.ndims(b::Buffer) = 1
Base.eltype{T}(b::Buffer{T}) = T

isnull{T}(b::Buffer) = (b.ptr == C_NULL)

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
