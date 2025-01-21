abstract type UnifiedMemory <: AbstractMemory end

## device buffer

"""
    UnifiedDeviceMemory

A buffer of device memory, owned by a specific device. Generally, may only be accessed by
the device that owns it.
"""
struct UnifiedDeviceMemory <: UnifiedMemory
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::Context
    device::Device
end

function device_alloc(
        ctx::Context, dev::Device, bytesize::Integer;
        alignment::Integer = 0, error_code::Ref{Int32} = Ref{Int32}(), properties::Tuple{Vararg{Symbol}} = ()
    )
    flags = 0
    if !isempty(properties)
        for i in properties
            if i == :wc
                flags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL
            else
                @warn "$i not recognized, ignoring flag. Valid optinos include `:wc`, `:ipd`, and `:iph`"
            end
        end
    end

    ptr = clDeviceMemAllocINTEL(ctx, dev, cl_mem_properties_intel[CL_MEM_ALLOC_FLAGS_INTEL, flags, 0], bytesize, alignment, error_code)

    @assert error_code[] == CL_SUCCESS
    #=
    @info ptr error_code[]
    result = Ref{UInt64}()
    @warn result
    success = clGetMemAllocInfoINTEL(ctx, ptr, CL_MEM_ALLOC_BASE_PTR_INTEL,
        sizeof(UInt64), result, C_NULL)

    @error success result
    @assert success == CL_SUCCESS
    =#
    return UnifiedDeviceMemory(reinterpret(CLPtr{Cvoid}, ptr), bytesize, ctx, dev)
end

Base.pointer(buf::UnifiedDeviceMemory) = buf.ptr
Base.sizeof(buf::UnifiedDeviceMemory) = buf.bytesize
context(buf::UnifiedDeviceMemory) = buf.context
device(buf::UnifiedDeviceMemory) = buf.device

Base.show(io::IO, buf::UnifiedDeviceMemory) =
    @printf(io, "UnifiedDeviceMemory(%s at %p)", Base.format_bytes(sizeof(buf)), pointer(buf))

Base.convert(::Type{CLPtr{T}}, buf::UnifiedDeviceMemory) where {T} =
    convert(CLPtr{T}, pointer(buf))


## host buffer

"""
    UnifiedHostMemory

A buffer of memory on the host. May be accessed by the host, and all devices within the
host driver. Frequently used as staging areas to transfer data to or from devices.
"""
struct UnifiedHostMemory <: UnifiedMemory
    ptr::Ptr{Cvoid}
    bytesize::Int
    context::Context
end

function host_alloc(
        ctx::Context, bytesize::Integer;
        alignment::Integer = 0, error_code::Ref{Int32} = Ref{Int32}(), properties::Tuple{Vararg{Symbol}} = ()
    )
    flags = 0
    if !isempty(properties)
        for i in properties
            if i == :wc
                flags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL
            else
                @warn "$i not recognized, ignoring flag. Valid optinos include `:wc`"
            end
        end
    end

    ptr = clHostMemAllocINTEL(ctx, cl_mem_properties_intel[CL_MEM_ALLOC_FLAGS_INTEL, flags, 0], bytesize, alignment, error_code)

    @assert error_code[] == CL_SUCCESS
    #=
    @info ptr error_code[]
    result = Ref{UInt64}()
    success = clGetMemAllocInfoINTEL(ctx, ptr, CL_MEM_ALLOC_BASE_PTR_INTEL,
        sizeof(UInt64), result, C_NULL)
    @assert success == CL_SUCCESS
    =#
    return UnifiedHostMemory(ptr, bytesize, ctx)
end

Base.pointer(buf::UnifiedHostMemory) = buf.ptr
Base.sizeof(buf::UnifiedHostMemory) = buf.bytesize
context(buf::UnifiedHostMemory) = buf.context
device(buf::UnifiedHostMemory) = nothing

Base.show(io::IO, buf::UnifiedHostMemory) =
    @printf(io, "UnifiedHostMemory(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::UnifiedHostMemory) where {T} =
    convert(Ptr{T}, pointer(buf))

Base.convert(::Type{CLPtr{T}}, buf::UnifiedHostMemory) where {T} =
    reinterpret(CLPtr{T}, pointer(buf))


## shared buffer

"""
    UnifiedSharedMemory

A managed buffer that is shared between the host and one or more devices.
"""
struct UnifiedSharedMemory <: UnifiedMemory
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::Context
    device::Union{Nothing, Device}
end

function shared_alloc(
        ctx::Context, dev::Device, bytesize::Integer;
        alignment::Integer = 0, error_code::Ref{Int32} = Ref{Int32}(), properties::Tuple{Vararg{Symbol}} = ()
    )
    flags = 0
    if !isempty(properties)
        if (:ipd in properties) && (:iph in properties)
            error("`properties` contains both `:ipd` and `:iph`, these flags are mutually exclusive.")
        end
        for i in properties
            if i == :wc
                flags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL
            elseif i == :ipd
                flags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL
            elseif i == :iph
                flags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL
            else
                @warn "$i not recognized, ignoring flag. Valid optinos include `:wc`, `:ipd`, and `:iph`"
            end
        end
    end

    ptr = clSharedMemAllocINTEL(ctx, dev, cl_mem_properties_intel[CL_MEM_ALLOC_FLAGS_INTEL, flags, 0], bytesize, alignment, error_code)

    @assert error_code[] == CL_SUCCESS
    #=
    @info ptr error_code[]
    result = Ref{UInt64}()
    @warn result
    success = clGetMemAllocInfoINTEL(ctx, ptr, CL_MEM_ALLOC_BASE_PTR_INTEL,
        sizeof(UInt64), result, C_NULL)

    @error success result
    @assert success == CL_SUCCESS
    =#
    return UnifiedSharedMemory(reinterpret(CLPtr{Cvoid}, ptr), bytesize, ctx, dev)
end

Base.pointer(buf::UnifiedSharedMemory) = buf.ptr
Base.sizeof(buf::UnifiedSharedMemory) = buf.bytesize
context(buf::UnifiedSharedMemory) = buf.context
device(buf::UnifiedSharedMemory) = buf.device

Base.show(io::IO, buf::UnifiedSharedMemory) =
    @printf(io, "UnifiedSharedMemory(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::UnifiedSharedMemory) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(buf)))

Base.convert(::Type{CLPtr{T}}, buf::UnifiedSharedMemory) where {T} =
    convert(CLPtr{T}, pointer(buf))


struct UnknownBuffer <: UnifiedMemory
    ptr::Ptr{Cvoid}
    bytesize::Int
    context::Context
end

Base.pointer(buf::UnknownBuffer) = buf.ptr
Base.sizeof(buf::UnknownBuffer) = buf.bytesize
context(buf::UnknownBuffer) = buf.context

Base.show(io::IO, buf::UnknownBuffer) =
    @printf(io, "UnknownBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

function enqueue_usm_memcpy(
        dst::Union{CLPtr, Ptr}, src::Union{CLPtr, Ptr}, nbytes::Integer; queu::CmdQueue = queue(), blocking::Bool = false,
        wait_for::Vector{Event} = Event[]
    )::Union{Event, Nothing}
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    return GC.@preserve wait_for begin
        com = (CL_NULL, C_NULL)
        ret_evt = (dst in com || src in com) ? C_NULL : Ref{cl_event}()
        clEnqueueMemcpyINTEL(queu, blocking, dst, src, nbytes, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

function enqueue_usm_memfill(
        dst::Union{CLPtr, Ptr}, pattern::Union{Ptr{T}, CLPtr{T}}, pattern_size::Integer, nbytes::Integer; queu::CmdQueue = queue(),
        wait_for::Vector{Event} = Event[]
    ) where {T}
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    return GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueMemFillINTEL(queu, dst, pattern, pattern_size, nbytes, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

function enqueue_usm_migratemem(
        ptr::Union{CLPtr, Ptr}, nbytes::Integer;
        queu::CmdQueue = queue(),
        wait_for::Vector{Event} = Event[],
        properties::Vector{Symbol} = Symbol[]
    )
    flag = 0
    if !isempty(properties)
        for i in properties
            if i == :host
                flag |= CL_MIGRATE_MEM_OBJECT_HOST
            elseif i == :undefined
                flag |= CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED
            else
                error("Invalid flag, the only allowed values for properties are :host and :undefined")
            end
        end
    end
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    return GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueMigrateMemINTEL(queu, ptr, nbytes, flag, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

function enqueue_usm_memadvise(
        ptr::Union{CLPtr, Ptr}, nbytes::Integer;
        queu::CmdQueue = queue(),
        wait_for::Vector{Event} = Event[],
        properties::Vector{Symbol} = Symbol[]
    )
    flag = 0
    if !isempty(properties)
        error("Invalid flag, no advise for memory according to USM yet, this parameter is for the future.")
    end
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    return GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueMemAdviseINTEL(queu, ptr, nbytes, flag, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end
