struct BufferDeviceMemory <: AbstractMemory
    buf::Union{Buffer, Nothing}
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::Context
end

BufferDeviceMemory() = BufferDeviceMemory(nothing, CL_NULL, 0, context())

function bda_alloc(bytesize::Integer;
        alignment::Integer = 0, device::Symbol = :rw, host::Symbol = :rw
    )

    # TODO: use alignment
    buf = Buffer(bytesize; device, host, device_private_address=true)

    addr = Ref{cl_mem_device_address_ext}()
    clGetMemObjectInfo(buf, CL_MEM_DEVICE_ADDRESS_EXT, sizeof(cl_mem_device_address_ext), addr, C_NULL)
    ptr = CLPtr{Cvoid}(addr[])
    @assert ptr != C_NULL
    return BufferDeviceMemory(buf, ptr, bytesize, context())
end

bda_free(mem::BufferDeviceMemory) = release(mem.buf)

Base.pointer(mem::BufferDeviceMemory) = mem.ptr
Base.sizeof(mem::BufferDeviceMemory) = mem.bytesize
context(mem::BufferDeviceMemory) = mem.context

Base.show(io::IO, mem::BufferDeviceMemory) =
    @printf(io, "BufferDeviceMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{Ptr{T}}, mem::BufferDeviceMemory) where {T} =
    convert(Ptr{T}, pointer(mem))

Base.convert(::Type{CLPtr{T}}, mem::BufferDeviceMemory) where {T} =
    reinterpret(CLPtr{T}, pointer(mem))

Base.convert(::Type{Buffer}, mem::BufferDeviceMemory) = mem.buf
