struct SharedVirtualMemory <: AbstractPointerMemory
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::Context
end

SharedVirtualMemory() = SharedVirtualMemory(CL_NULL, 0, context())

function svm_alloc(bytesize::Integer;
        alignment::Integer = 0, access::Symbol = :rw, fine_grained = false
    )

    flags = if access == :rw
        CL_MEM_READ_WRITE
    elseif access == :r
        CL_MEM_READ_ONLY
    elseif access == :w
        CL_MEM_WRITE_ONLY
    else
        throw(ArgumentError("Invalid access type"))
    end

    if fine_grained
        flags |= CL_MEM_SVM_FINE_GRAIN_BUFFER
    end

    ptr = clSVMAlloc(context(), flags, bytesize, alignment)
    @assert ptr != C_NULL

    # JuliaGPU/OpenCL.jl#252: uninitialized SVM memory doesn't work on Intel
    if platform().name == "Intel(R) OpenCL Graphics"
        enqueue_svm_fill(ptr, UInt8(0), bytesize)
    end

    return SharedVirtualMemory(ptr, bytesize, context())
end

svm_free(mem::SharedVirtualMemory) = clSVMFree(context(mem), mem)

Base.pointer(mem::SharedVirtualMemory) = mem.ptr
Base.sizeof(mem::SharedVirtualMemory) = mem.bytesize
context(mem::SharedVirtualMemory) = mem.context

Base.show(io::IO, mem::SharedVirtualMemory) =
    @printf(io, "SharedVirtualMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{Ptr{T}}, mem::SharedVirtualMemory) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(mem)))

Base.convert(::Type{CLPtr{T}}, mem::SharedVirtualMemory) where {T} =
    reinterpret(CLPtr{T}, pointer(mem))


## memory operations

# these generally only make sense for coarse-grained SVM buffers;
# fine-grained buffers can just be used directly.

# copy from and to SVM buffers
function enqueue_svm_copy(
        dst::Union{Ptr, CLPtr}, src::Union{Ptr, CLPtr}, nbytes::Integer; queue::CmdQueue = queue(), blocking::Bool = false,
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
