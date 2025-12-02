abstract type UnifiedMemory <: AbstractPointerMemory end

function usm_free(mem::UnifiedMemory; blocking::Bool = false)
    if blocking
        clMemBlockingFreeINTEL(context(mem), mem)
    else
        clMemFreeINTEL(context(mem), mem)
    end
end


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
end

UnifiedDeviceMemory() = UnifiedDeviceMemory(CL_NULL, 0, context())

function device_alloc(bytesize::Integer;
        alignment::Integer = 0, write_combined::Bool = false
    )

    flags = 0
    if write_combined
        flags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL
    end

    error_code = Ref{Cint}()
    props = cl_mem_properties_intel[CL_MEM_ALLOC_FLAGS_INTEL, flags, 0]
    ptr = clDeviceMemAllocINTEL(context(), device(), props, bytesize, alignment, error_code)
    if error_code[] != CL_SUCCESS
        throw(CLError(error_code[]))
    end

    return UnifiedDeviceMemory(ptr, bytesize, context())
end

Base.pointer(mem::UnifiedDeviceMemory) = mem.ptr
Base.sizeof(mem::UnifiedDeviceMemory) = mem.bytesize
context(mem::UnifiedDeviceMemory) = mem.context

Base.show(io::IO, mem::UnifiedDeviceMemory) =
    @printf(io, "UnifiedDeviceMemory(%s at %p)", Base.format_bytes(sizeof(mem)), pointer(mem))

Base.convert(::Type{CLPtr{T}}, mem::UnifiedDeviceMemory) where {T} =
    convert(CLPtr{T}, pointer(mem))


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

UnifiedHostMemory() = UnifiedHostMemory(C_NULL, 0, context())

function host_alloc(bytesize::Integer;
        alignment::Integer = 0, write_combined::Bool = false
    )

    flags = 0
    if write_combined
        flags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL
    end

    error_code = Ref{Cint}()
    props = cl_mem_properties_intel[CL_MEM_ALLOC_FLAGS_INTEL, flags, 0]
    ptr = clHostMemAllocINTEL(context(), props, bytesize, alignment, error_code)
    if error_code[] != CL_SUCCESS
        throw(CLError(error_code[]))
    end

    return UnifiedHostMemory(ptr, bytesize, context())
end

Base.pointer(mem::UnifiedHostMemory) = mem.ptr
Base.sizeof(mem::UnifiedHostMemory) = mem.bytesize
context(mem::UnifiedHostMemory) = mem.context

Base.show(io::IO, mem::UnifiedHostMemory) =
    @printf(io, "UnifiedHostMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{Ptr{T}}, mem::UnifiedHostMemory) where {T} =
    convert(Ptr{T}, pointer(mem))


## shared buffer

"""
    UnifiedSharedMemory

A managed buffer that is shared between the host and one or more devices.
"""
struct UnifiedSharedMemory <: UnifiedMemory
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::Context
end

UnifiedSharedMemory() = UnifiedSharedMemory(CL_NULL, 0, context())

function shared_alloc(bytesize::Integer;
        alignment::Integer = 0, write_combined = false, placement = nothing
    )

    flags = 0
    if write_combined
        flags |= CL_MEM_ALLOC_WRITE_COMBINED_INTEL
    end
    if placement !== nothing
        if placement == :host
            flags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL
        elseif placement == :device
            flags |= CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL
        else
            error("Invalid placement, the only allowed values for placement are :host and :device")
        end
    end

    error_code = Ref{Cint}()
    props = cl_mem_properties_intel[CL_MEM_ALLOC_FLAGS_INTEL, flags, 0]
    ptr = clSharedMemAllocINTEL(context(), device(), props, bytesize, alignment, error_code)
    if error_code[] != CL_SUCCESS
        throw(CLError(error_code[]))
    end

    return UnifiedSharedMemory(ptr, bytesize, context())
end

Base.pointer(mem::UnifiedSharedMemory) = mem.ptr
Base.sizeof(mem::UnifiedSharedMemory) = mem.bytesize
context(mem::UnifiedSharedMemory) = mem.context

Base.show(io::IO, mem::UnifiedSharedMemory) =
    @printf(io, "UnifiedSharedMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{Ptr{T}}, mem::UnifiedSharedMemory) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(mem)))

Base.convert(::Type{CLPtr{T}}, mem::UnifiedSharedMemory) where {T} =
    convert(CLPtr{T}, pointer(mem))


## memory operations

function enqueue_usm_copy(
        dst::Union{CLPtr, Ptr}, src::Union{CLPtr, Ptr}, nbytes::Integer; queue::CmdQueue = queue(), blocking::Bool = false,
        wait_for::Vector{Event} = Event[]
    )::Union{Event, Nothing}
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    return GC.@preserve wait_for begin
        com = (CL_NULL, C_NULL)
        ret_evt = (dst in com || src in com) ? C_NULL : Ref{cl_event}()
        clEnqueueMemcpyINTEL(queue, blocking, dst, src, nbytes, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

# fill a buffer with a pattern, returning an event
function enqueue_usm_fill(ptr::Union{Ptr, CLPtr}, pattern::T, N::Integer;
                          queue::CmdQueue = queue(),
                          wait_for::Vector{Event}=Event[]) where {T}
    nbytes = N * sizeof(T)
    pattern_size = sizeof(T)
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueMemFillINTEL(queue, ptr, Ref(pattern),
                              pattern_size, nbytes,
                              n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

function enqueue_usm_migrate(
        ptr::Union{CLPtr, Ptr}, nbytes::Integer;
        queue::CmdQueue = queue(),
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
        clEnqueueMigrateMemINTEL(queue, ptr, nbytes, flag, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

function enqueue_usm_advise(
        ptr::Union{CLPtr, Ptr}, nbytes::Integer;
        queue::CmdQueue = queue(),
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
        clEnqueueMemAdviseINTEL(queue, ptr, nbytes, flag, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end
