# OpenCL.Buffer

struct Buffer <: AbstractMemoryObject
    id::cl_mem
    ptr::Union{Nothing,CLPtr{Cvoid}}
    bytesize::Int
    context::Context
end

Buffer() = Buffer(C_NULL, CL_NULL, 0, context())

Base.pointer(buf::Buffer) = @something buf.ptr error("Conversion of a buffer to a pointer is not supported by this device")
Base.sizeof(buf::Buffer) = buf.bytesize
context(buf::Buffer) = buf.context


## constructors

# for internal use
function Buffer(sz::Int, flags::Integer, hostbuf=nothing;
                device=:rw, host=:rw, device_private_address=bda_supported(cl.device()))
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
    properties = cl_mem_properties[]
    if device_private_address
        append!(properties, [CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT, CL_TRUE])
    end
    mem_id = if isempty(properties)
        clCreateBuffer(context(), flags, sz, something(hostbuf, C_NULL), err_code)
    else
        push!(properties, 0)
        clCreateBufferWithProperties(context(), properties, flags, sz, something(hostbuf, C_NULL), err_code)
    end
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end

    ptr = if device_private_address
        ptr_ref = Ref{cl_mem_device_address_ext}()
        clGetMemObjectInfo(mem_id, CL_MEM_DEVICE_ADDRESS_EXT, sizeof(cl_mem_device_address_ext), ptr_ref, C_NULL)
        CLPtr{Cvoid}(ptr_ref[])
    else
        nothing
    end

    return Buffer(mem_id, ptr, sz, context())
end

# allocated buffer
function Buffer(sz::Integer; host_accessible=false, kwargs...)
    flags = host_accessible ? CL_MEM_ALLOC_HOST_PTR : 0
    Buffer(sz, flags, nothing; kwargs...)
end

# from host memory
function Buffer(hostbuf::Array; copy::Bool=true, kwargs...)
    flags = copy ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR
    Buffer(sizeof(hostbuf), flags, hostbuf; kwargs...)
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
function enqueue_map(buf::Buffer, offset::Integer, nbytes::Int, flags=:rw;
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
        ptr = clEnqueueMapBuffer(queue(), buf, blocking, flags, offset, nbytes,
                                 n_evts, evt_ids, ret_evt, status)
        if status[] != CL_SUCCESS
            throw(CLError(status[]))
        end

        return ptr, Event(ret_evt[])
    end
end
enqueue_map(buf::Buffer, nbytes::Int, flags=:rw; kwargs...) =
    enqueue_map(buf, 0, nbytes, flags; kwargs...)

# unmap a buffer, return an event
function enqueue_unmap(buf::Buffer, ptr::Ptr; wait_for::Vector{Event}=Event[])
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueUnmapMemObject(queue(), buf, ptr, n_evts, evt_ids, ret_evt)
        return Event(ret_evt[])
    end
end

# fill a buffer with a pattern, returning an event
function enqueue_fill(buf::Buffer, offset::Integer, pattern::T, N::Integer;
                      wait_for::Vector{Event}=Event[]) where {T}
    nbytes = N * sizeof(T)
    nbytes_pattern = sizeof(T)
    @assert nbytes_pattern > 0
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve begin
        ret_evt = Ref{cl_event}()
        clEnqueueFillBuffer(queue(), buf, [pattern],
                            nbytes_pattern, offset, nbytes,
                            n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end
enqueue_fill(buf::Buffer, pattern, N::Integer) = enqueue_fill(buf, 0, pattern, N)
