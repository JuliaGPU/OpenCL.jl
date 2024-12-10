# Raw memory management

export device_alloc, host_alloc, shared_alloc, free#, properties, lookup_alloc

#
# untyped buffers
#

abstract type AbstractBuffer end

Base.convert(T::Type{<:Union{Ptr,CLPtr}}, buf::AbstractBuffer) =
    throw(ArgumentError("Illegal conversion of a $(typeof(buf)) to a $T"))

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(P::Type{<:Union{Ptr,CLPtr}}, buf::AbstractBuffer) = convert(P, buf)

function free(buf::AbstractBuffer; blocking = false)
    ctx = context(buf)
    freefun = if blocking
        clMemBlockingFreeINTEL
    else 
        clMemFreeINTEL
    end
    success = freefun(ctx, Ptr{Nothing}(UInt(buf.ptr)))
	@assert success == cl.CL_SUCCESS
	return success
end

## device buffer

"""
    DeviceBuffer

A buffer of device memory, owned by a specific device. Generally, may only be accessed by
the device that owns it.
"""
struct DeviceBuffer <: AbstractBuffer
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::cl.Context
    device::cl.Device
end

function device_alloc(ctx::cl.Context, dev::cl.Device, bytesize::Integer;
                      alignment::Integer=0, error_code::Ref{Int32}=Ref{Int32}(), properties::Tuple{Vararg{Symbol}}=())
	flags = 0
	if !isempty(properties)
		for i in properties
			if i == :wc
				flags |= cl.CL_MEM_ALLOC_WRITE_COMBINED_INTEL
			else
				@warn "$i not recognized, ignoring flag. Valid optinos include `:wc`, `:ipd`, and `:iph`"
			end
		end
	end
	
	ptr = clDeviceMemAllocINTEL(ctx, dev,cl.cl_mem_properties_intel[cl.CL_MEM_ALLOC_FLAGS_INTEL, flags, 0], bytesize, alignment, error_code)
	
	@assert error_code[] == cl.CL_SUCCESS
	#=
	@info ptr error_code[]
	result = Ref{UInt64}()
	@warn result
	success = clGetMemAllocInfoINTEL(ctx, ptr, cl.CL_MEM_ALLOC_BASE_PTR_INTEL, 
        sizeof(UInt64), result, C_NULL)
	
	@error success result
	@assert success == cl.CL_SUCCESS
    =#
	return DeviceBuffer(reinterpret(CLPtr{Cvoid}, ptr), bytesize, ctx, dev)
end

Base.pointer(buf::DeviceBuffer) = buf.ptr
Base.sizeof(buf::DeviceBuffer) = buf.bytesize
context(buf::DeviceBuffer) = buf.context
device(buf::DeviceBuffer) = buf.device

Base.show(io::IO, buf::DeviceBuffer) =
    @printf(io, "DeviceBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), pointer(buf))

Base.convert(::Type{CLPtr{T}}, buf::DeviceBuffer) where {T} =
    convert(CLPtr{T}, pointer(buf))


## host buffer

"""
    HostBuffer

A buffer of memory on the host. May be accessed by the host, and all devices within the
host driver. Frequently used as staging areas to transfer data to or from devices.

Note that these buffers need to be made resident to the device, e.g., by using the
ZE_KERNEL_FLAG_FORCE_RESIDENCY module flag, the ZE_KERNEL_SET_ATTR_INDIRECT_HOST_ACCESS
kernel attribute, or by calling zeDeviceMakeMemoryResident.
"""
struct HostBuffer <: AbstractBuffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    context::cl.Context
end

function host_alloc(ctx::cl.Context, bytesize::Integer;
                      alignment::Integer=0, error_code::Ref{Int32}=Ref{Int32}(), properties::Tuple{Vararg{Symbol}}=())
	flags = 0
	if !isempty(properties)
		for i in properties
			if i == :wc
				flags |= cl.CL_MEM_ALLOC_WRITE_COMBINED_INTEL
			else
				@warn "$i not recognized, ignoring flag. Valid optinos include `:wc`"
			end
		end
	end
	
	ptr = clDeviceMemAllocINTEL(ctx, cl.cl_mem_properties_intel[cl.CL_MEM_ALLOC_FLAGS_INTEL, flags, 0], bytesize, alignment, error_code)
	
	@assert error_code[] == cl.CL_SUCCESS
	#=
	@info ptr error_code[]
	result = Ref{UInt64}()
	@warn result
	success = clGetMemAllocInfoINTEL(ctx, ptr, cl.CL_MEM_ALLOC_BASE_PTR_INTEL, 
        sizeof(UInt64), result, C_NULL)
	
	@error success result
	@assert success == cl.CL_SUCCESS
    =#
	return HostBuffer(ptr, bytesize, ctx)
end

#=
function host_alloc(ctx::CLContext, bytesize::Integer, alignment::Integer=1;
                    flags=0)
    desc_ref = Ref(ze_host_mem_alloc_desc_t(; flags))

    ptr_ref = Ref{Ptr{Cvoid}}()
    zeMemAllocHost(ctx, desc_ref, bytesize, alignment, ptr_ref)

    return HostBuffer(ptr_ref[], bytesize, ctx)
end
=#

Base.pointer(buf::HostBuffer) = buf.ptr
Base.sizeof(buf::HostBuffer) = buf.bytesize
context(buf::HostBuffer) = buf.context
device(buf::HostBuffer) = nothing

Base.show(io::IO, buf::HostBuffer) =
    @printf(io, "HostBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::HostBuffer) where {T} =
    convert(Ptr{T}, pointer(buf))

Base.convert(::Type{CLPtr{T}}, buf::HostBuffer) where {T} =
    reinterpret(CLPtr{T}, pointer(buf))


## shared buffer

"""
    SharedBuffer

A managed buffer that is shared between the host and one or more devices.
"""
struct SharedBuffer <: AbstractBuffer
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::cl.Context
    device::Union{Nothing,cl.Device}
end

function shared_alloc(ctx::cl.Context, dev::cl.Device, bytesize::Integer;
                      alignment::Integer=0, error_code::Ref{Int32}=Ref{Int32}(), properties::Tuple{Vararg{Symbol}}=())
	flags = 0
	if !isempty(properties)
		if (:ipd in properties) && (:iph in properties)
			error("`properties` contains both `:ipd` and `:iph`, these flags are mutually exclusive.")
		end
		for i in properties
			if i == :wc
				flags |= cl.CL_MEM_ALLOC_WRITE_COMBINED_INTEL
			elseif i == :ipd
				flags |= cl.CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL
			elseif i == :iph
				flags |= cl.CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL
			else
				@warn "$i not recognized, ignoring flag. Valid optinos include `:wc`, `:ipd`, and `:iph`"
			end
		end
	end
	
	ptr = clSharedMemAllocINTEL(ctx, dev, cl.cl_mem_properties_intel[cl.CL_MEM_ALLOC_FLAGS_INTEL, flags, 0], bytesize, alignment, error_code)
	
	@assert error_code[] == cl.CL_SUCCESS
	#=
	@info ptr error_code[]
	result = Ref{UInt64}()
	@warn result
	success = clGetMemAllocInfoINTEL(ctx, ptr, cl.CL_MEM_ALLOC_BASE_PTR_INTEL, 
        sizeof(UInt64), result, C_NULL)
	
	@error success result
	@assert success == cl.CL_SUCCESS
    =#
	return SharedBuffer(reinterpret(CLPtr{Cvoid}, ptr), bytesize, ctx, dev)
end

#=
function shared_alloc(ctx::CLContext, dev::Union{Nothing,CLDevice},
                      bytesize::Integer, alignment::Integer=1; host_flags=0,
                      device_flags=0, ordinal::Integer=0)
    relaxed_allocation_ref = Ref(ze_relaxed_allocation_limits_exp_desc_t(;
        flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE,
    ))
    GC.@preserve relaxed_allocation_ref begin
        device_desc_ref = if dev !== nothing && bytesize > properties(dev).maxMemAllocSize
            pNext = Base.unsafe_convert(Ptr{Cvoid}, relaxed_allocation_ref)
            Ref(ze_device_mem_alloc_desc_t(; flags=device_flags, ordinal, pNext))
        else
            Ref(ze_device_mem_alloc_desc_t(; flags=device_flags, ordinal))
        end
        host_desc_ref = Ref(ze_host_mem_alloc_desc_t(; flags=host_flags))

        ptr_ref = Ref{Ptr{Cvoid}}()
        zeMemAllocShared(ctx, device_desc_ref, host_desc_ref, bytesize, alignment,
                        something(dev, C_NULL), ptr_ref)

        return SharedBuffer(reinterpret(CLPtr{Cvoid}, ptr_ref[]), bytesize, ctx, dev)
    end
end
=#

Base.pointer(buf::SharedBuffer) = buf.ptr
Base.sizeof(buf::SharedBuffer) = buf.bytesize
context(buf::SharedBuffer) = buf.context
device(buf::SharedBuffer) = buf.device

Base.show(io::IO, buf::SharedBuffer) =
    @printf(io, "SharedBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::SharedBuffer) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(buf)))

Base.convert(::Type{CLPtr{T}}, buf::SharedBuffer) where {T} =
    convert(CLPtr{T}, pointer(buf))


## properties
#=
function properties(buf::AbstractBuffer)
    props_ref = Ref(ze_memory_allocation_properties_t())
    dev_ref = Ref(ze_device_handle_t())
    zeMemGetAllocProperties(buf.context, pointer(buf), props_ref, dev_ref)

	result = Ref{}()
	success = clGetMemAllocInfoINTEL(ctx, ptr, cl.CL_MEM_ALLOC_BASE_PTR_INTEL, 
        sizeof(UInt64), result, C_NULL)

    
    props = props_ref[]
    return (
        device=cl.Device(dev_ref[], buf.context.driver),
        type=props.type,
        id=props.id,
    )
end
=#
struct UnknownBuffer <: AbstractBuffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    context::cl.Context
end

Base.pointer(buf::UnknownBuffer) = buf.ptr
Base.sizeof(buf::UnknownBuffer) = buf.bytesize
context(buf::UnknownBuffer) = buf.context

Base.show(io::IO, buf::UnknownBuffer) =
    @printf(io, "UnknownBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

#=
function lookup_alloc(ctx::cl.Context, ptr::Union{Ptr,CLPtr})
    base_ref = Ref{Ptr{Cvoid}}()
    bytesize_ref = Ref{Csize_t}()
    
    zeMemGetAddressRange(ctx, ptr, base_ref, bytesize_ref)

    buf = UnknownBuffer(base_ref[], bytesize_ref[], ctx)
    props = properties(buf)
    return if props.type == ZE_MEMORY_TYPE_HOST
        HostBuffer(pointer(buf), sizeof(buf), ctx)
    elseif props.type == ZE_MEMORY_TYPE_DEVICE
        DeviceBuffer(reinterpret(CLPtr{Cvoid}, pointer(buf)), sizeof(buf), ctx, props.device)
    elseif props.type == ZE_MEMORY_TYPE_SHARED
        SharedBuffer(reinterpret(CLPtr{Cvoid}, pointer(buf)), sizeof(buf), ctx, props.device)
    else
        buf
    end
end
=#
