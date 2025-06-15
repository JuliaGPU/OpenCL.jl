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

enqueue_bda_copy(dst::Ptr, src::cl_mem, nbytes; kwargs...) =
    enqueue_read(dst, src, nbytes; kwargs...)

enqueue_bda_copy(dst::cl_mem, src::Ptr, nbytes; kwargs...) =
    enqueue_write(dst, src, nbytes; kwargs...)

enqueue_bda_copy(dst::cl_mem, src::cl_mem, nbytes; kwargs...) =
    enqueue_copy(dst, src, nbytes; kwargs...)
    
enqueue_bda_copy(dst::Ptr, dst_off::Int, src::cl_mem, src_off::Int, nbytes; kwargs...) =
    enqueue_read(dst, src, src_off, nbytes; kwargs...)
    
enqueue_bda_copy(dst::cl_mem, dst_off::Int, src::Ptr, src_off::Int, nbytes; kwargs...) =
    enqueue_write(dst, dst_off, src, nbytes; kwargs...)

enqueue_bda_copy(dst::cl_mem, dst_off::Int, src::cl_mem, src_off::Int, nbytes; kwargs...) =
    enqueue_copy(dst, dst_off, src, src_off, nbytes; kwargs...)
