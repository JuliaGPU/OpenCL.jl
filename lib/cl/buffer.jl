# OpenCL Memory Object

abstract type AbstractMemory <: CLObject end

#This should be implemented by all subtypes
# type MemoryType <: AbstractMemory
#     id::cl_mem
#     ...
# end

# for passing buffers to OpenCL APIs: use the underlying handle
Base.unsafe_convert(::Type{cl_mem}, mem::AbstractMemory) = mem.id

# for passing buffers to kernels: keep the buffer, it's handled by `cl.set_arg!`
Base.unsafe_convert(::Type{<:Ptr}, mem::AbstractMemory) = mem

Base.sizeof(mem::AbstractMemory) = mem.size

context(mem::AbstractMemory) = mem.context

function Base.getproperty(mem::AbstractMemory, s::Symbol)
    if s == :context
        param = Ref{cl_context}()
        clGetMemObjectInfo(mem, CL_MEM_CONTEXT, sizeof(cl_context), param, C_NULL)
        return Context(param[], retain = true)
    elseif s == :mem_type
        result = Ref{cl_mem_object_type}()
        clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(cl_mem_object_type), result, C_NULL)
        return result[]
    elseif s == :mem_flags
        result = Ref{cl_mem_flags}()
        clGetMemObjectInfo(mem, CL_MEM_FLAGS, sizeof(cl_mem_flags), result, C_NULL)
        mf = result[]
        flags = Symbol[]
        if (mf & CL_MEM_READ_WRITE) != 0
            push!(flags, :rw)
        end
        if (mf & CL_MEM_WRITE_ONLY) != 0
            push!(flags, :w)
        end
        if (mf & CL_MEM_READ_ONLY) != 0
            push!(flags, :r)
        end
        if (mf & CL_MEM_USE_HOST_PTR) != 0
            push!(flags, :use)
        end
        if (mf & CL_MEM_ALLOC_HOST_PTR) != 0
            push!(flags, :alloc)
        end
        if (mf & CL_MEM_COPY_HOST_PTR) != 0
            push!(flags, :copy)
        end
        return tuple(flags...)
    elseif s == :size
        result = Ref{Csize_t}()
        clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(Csize_t), result, C_NULL)
        return result[]
    elseif s == :reference_count
        result = Ref{Cuint}()
        clGetMemObjectInfo(mem, CL_MEM_REFERENCE_COUNT, sizeof(Cuint), result, C_NULL)
        return Int(result[])
    elseif s == :map_count
        result = Ref{Cuint}()
        clGetMemObjectInfo(mem, CL_MEM_MAP_COUNT, sizeof(Cuint), result, C_NULL)
        return Int(result[])
    else
        return getfield(mem, s)
    end
end

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)

# OpenCL.Buffer

mutable struct Buffer{T} <: AbstractMemory
    const id::cl_mem
    const len::Int

    function Buffer{T}(mem_id::cl_mem, len::Integer; retain::Bool=false) where {T}
        buff = new{T}(mem_id, len)
        retain && clRetainMemObject(buff)
        finalizer(clReleaseMemObject, buff)
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
function enqueue_read(dst::Ptr, src::Buffer, src_off::Int, nbytes::Int;
                      blocking::Bool=false, wait_for::Vector{Event}=Event[])
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueReadBuffer(queue(), src, blocking, src_off, nbytes, dst,
                            n_evts, evt_ids, ret_evt)
        @return_nanny_event(ret_evt[], dst)
    end
end
enqueue_read(dst::Ptr, src::Buffer, nbytes; kwargs...) =
    enqueue_read(dst, src, 0, nbytes; kwargs...)

# writing from host array to buffer, return an event
function enqueue_write(dst::Buffer, dst_off::Int, src::Ptr, nbytes::Int;
                       blocking::Bool=false, wait_for::Vector{Event}=Event[])
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueWriteBuffer(queue(), dst, blocking, dst_off, nbytes, src,
                             n_evts, evt_ids, ret_evt)
        @return_nanny_event(ret_evt[], dst)
    end
end
enqueue_write(dst::Buffer, src::Ptr, nbytes; kwargs...) =
    enqueue_write(dst, 0, src, nbytes; kwargs...)

# copying between two buffers, return an event
function enqueue_copy(dst::Buffer, dst_off::Int, src::Buffer, src_off::Int,
                      nbytes::Int; blocking::Bool=false,
                      wait_for::Vector{Event}=Event[])
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueCopyBuffer(queue(), src, dst, src_off, dst_off, nbytes,
                            n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end
enqueue_copy(dst::Buffer, src::Buffer, N; kwargs...) =
    enqueue_copy(dst, 0, src, 0, N; kwargs...)

# map a buffer into the host address space, returning a pointer and an event
function enqueue_map(b::Buffer, offset::Integer, nbytes::Int, flags=:rw;
                     blocking::Bool=false, wait_for::Vector{Event}=Event[])
    flags = if flags == :rw
        CL_MAP_READ | CL_MAP_WRITE
    elseif flags == :r
        CL_MAP_READ
    elseif flags == :w
        CL_MAP_WRITE
    else
        throw(ArgumentError("enqueue_unmap can have flags of :r, :w, or :rw, got :$flags"))
    end

    ret_evt = Ref{cl_event}()
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        status  = Ref{Cint}()
        ptr = clEnqueueMapBuffer(queue(), b, blocking, flags, offset, nbytes,
                                 n_evts, evt_ids, ret_evt, status)
        if status[] != CL_SUCCESS
            throw(CLError(status[]))
        end

        return ptr, Event(ret_evt[])
    end
end
enqueue_map(b::Buffer, nbytes::Int, flags=:rw; kwargs...) =
    enqueue_map(b, 0, nbytes, flags; kwargs...)

# unmap a buffer, return an event
function enqueue_unmap(b::Buffer, ptr::Ptr; wait_for::Vector{Event}=Event[])
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueUnmapMemObject(queue(), b, ptr, n_evts, evt_ids, ret_evt)
        return Event(ret_evt[])
    end
end

# fill a buffer with a pattern, returning an event
function enqueue_fill(b::Buffer, offset::Integer, pattern::T, N::Integer;
                      wait_for::Vector{Event}=Event[]) where {T}
    nbytes = N * sizeof(T)
    nbytes_pattern = sizeof(T)
    @assert nbytes_pattern > 0
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve begin
        ret_evt = Ref{cl_event}()
        clEnqueueFillBuffer(queue(), b, [pattern],
                            nbytes_pattern, offset, nbytes,
                            n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end
enqueue_fill(b::Buffer, pattern, N::Integer) = enqueue_fill(b, 0, pattern, N)
