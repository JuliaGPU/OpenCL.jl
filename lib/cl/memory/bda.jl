struct BufferDeviceMemory <: AbstractMemory
    id::cl_mem
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::Context
end

BufferDeviceMemory() = BufferDeviceMemory(C_NULL, CL_NULL, 0, context())

function bda_alloc(bytesize::Integer;
        alignment::Integer = 0, device_access::Symbol = :rw, host_access::Symbol = :rw
    )
    bytesize == 0 && return BufferDeviceMemory()

    flags = if device_access == :rw
        CL_MEM_READ_WRITE
    elseif device_access == :r
        CL_MEM_READ_ONLY
    elseif device_access == :w
        CL_MEM_WRITE_ONLY
    else
        throw(ArgumentError("Invalid access type"))
    end

    if host_access == :rw
        # nothing to do
    elseif host_access == :r
        flags |= CL_MEM_HOST_READ_ONLY
    elseif host_access == :w
        flags |= CL_MEM_HOST_WRITE_ONLY
    elseif host_access == :none
        flags |= CL_MEM_HOST_NO_ACCESS
    else
        throw(ArgumentError("Host access flag must be one of :rw, :r, or :w"))
    end

    
    err_code = Ref{Cint}()
    properties = cl_mem_properties[CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT, CL_TRUE, 0]
    mem_id = clCreateBufferWithProperties(context(), properties, flags, bytesize, C_NULL, err_code)
    addr = Ref{cl_mem_device_address_ext}()
    clGetMemObjectInfo(mem_id, CL_MEM_DEVICE_ADDRESS_EXT, sizeof(cl_mem_device_address_ext), addr, C_NULL)
    ptr = CLPtr{Cvoid}(addr[])
    @assert ptr != C_NULL
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end
    return BufferDeviceMemory(mem_id, ptr, bytesize, context())
end

function bda_free(buf::BufferDeviceMemory)
    if sizeof(buf) != 0
        clReleaseMemObject(buf.id)
    end
    return
end

Base.pointer(buf::BufferDeviceMemory) = buf.ptr
Base.sizeof(buf::BufferDeviceMemory) = buf.bytesize
context(buf::BufferDeviceMemory) = buf.context

Base.show(io::IO, buf::BufferDeviceMemory) =
    @printf(io, "BufferDeviceMemory(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::BufferDeviceMemory) where {T} =
    convert(Ptr{T}, pointer(buf))

Base.convert(::Type{CLPtr{T}}, buf::BufferDeviceMemory) where {T} =
    reinterpret(CLPtr{T}, pointer(buf))

#=
## memory operations

# these generally only make sense for coarse-grained SVM buffers;
# fine-grained buffers can just be used directly.

# copy from and to SVM buffers
function enqueue_svm_copy(
        
        dst::Union{Ptr, CLPtr}, src::Union{Ptr, CLPtr}, nbytes::Integer; queue::CmdQueue = queue(), bloc, C_NULL)ing::Bool = false,
        wait_for::Vector{Event} = Event[]
    )
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    return GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMemcpy(queue, blocking, dst, src, nbytes, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

# map an SVM buffer into the host address space, returning an event
function enqueue_svm_map(
        ptr::Union{Ptr, CLPtr}, nbytes::Integer, flags = :rw; queue::CmdQueue = queue(), blocking::Bool = false,
        wait_for::Vector{Event} = Event[]
    )
    flags = if flags == :rw
        CL_MAP_READ | CL_MAP_WRITE
    elseif flags == :r
        CL_MAP_READ
    elseif flags == :w
        CL_MAP_WRITE
    else
        throw(ArgumentError("enqueue_unmap can have flags of :r, :w, or :rw, got :$flags"))
    end
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMap(
            queue, blocking, flags, ptr, nbytes,
            n_evts, evt_ids, ret_evt
        )

        return Event(ret_evt[])
    end
end

# unmap a buffer, returning an event
function enqueue_svm_unmap(ptr::Union{Ptr, CLPtr}; queue::CmdQueue = queue(), wait_for::Vector{Event} = Event[])
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMUnmap(queue, ptr, n_evts, evt_ids, ret_evt)
        return Event(ret_evt[])
    end
end

# fill a buffer with a pattern, returning an event
function enqueue_svm_fill(ptr::Union{Ptr, CLPtr}, pattern::T, N::Integer;
                          wait_for::Vector{Event}=Event[]) where {T}
    nbytes = N * sizeof(T)
    nbytes == 0 && return
    pattern_size = sizeof(T)
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMemFill(queue(), ptr, Ref(pattern),
                            pattern_size, nbytes,
                            n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end
=#
