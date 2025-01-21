struct SharedVirtualMemory <: AbstractMemory
    ptr::CLPtr{Cvoid}
    bytesize::Int
    context::Context
end

function svm_alloc(
        ctx::Context, bytesize::Integer;
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

    ptr = clSVMAlloc(ctx, flags, bytesize, alignment)

    @assert ptr != C_NULL
    #=
    # JuliaGPU/OpenCL.jl#252: uninitialized SVM memory doesn't work on Intel
    if platform().name == "Intel(R) OpenCL Graphics"
        len > 0 && enqueue_svm_fill(ptr, zero(T), len)
    end
    =#
    return SharedVirtualMemory(reinterpret(CLPtr{Cvoid}, ptr), bytesize, ctx)
end

Base.pointer(buf::SharedVirtualMemory) = buf.ptr
Base.sizeof(buf::SharedVirtualMemory) = buf.bytesize
context(buf::SharedVirtualMemory) = buf.context
device(::SharedVirtualMemory) = nothing

Base.show(io::IO, buf::SharedVirtualMemory) =
    @printf(io, "SharedVirtualMemory(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::SharedVirtualMemory) where {T} =
    convert(Ptr{T}, pointer(buf))

Base.convert(::Type{CLPtr{T}}, buf::SharedVirtualMemory) where {T} =
    reinterpret(CLPtr{T}, pointer(buf))


## memory operations

# these generally only make sense for coarse-grained SVM buffers;
# fine-grained buffers can just be used directly.

# copy from and to SVM buffers
function enqueue_svm_memcpy(
        dst::Union{Ptr, CLPtr}, src::Union{Ptr, CLPtr}, nbytes::Integer; queu::CmdQueue = queue(), blocking::Bool = false,
        wait_for::Vector{Event} = Event[]
    )
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    return GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMemcpy(queu, blocking, dst, src, nbytes, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end

# map an SVM buffer into the host address space, returning an event
function enqueue_svm_map(
        ptr::Union{Ptr, CLPtr}, nbytes::Integer, flags = :rw; queu::CmdQueue = queue(), blocking::Bool = false,
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
            queu, blocking, flags, ptr, nbytes,
            n_evts, evt_ids, ret_evt
        )

        return Event(ret_evt[])
    end
end

# unmap a buffer, returning an event
function enqueue_svm_unmap(ptr::Union{Ptr, CLPtr}; queu::CmdQueue = queue(), wait_for::Vector{Event} = Event[])
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMUnmap(queu, ptr, n_evts, evt_ids, ret_evt)
        return Event(ret_evt[])
    end
end

# fill a buffer with a pattern, returning an event
function enqueue_svm_fill(ptr::Union{Ptr, CLPtr}, pattern::Union{Ptr{T}, CLPtr{T}}, pattern_size::Integer, nbytes::Integer; queu::CmdQueue = queue(), wait_for::Vector{Event} = Event[]) where {T}
    @assert pattern_size > 0
    n_evts = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    return GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMemFill(
            queu, ptr, pattern,
            pattern_size, nbytes,
            n_evts, evt_ids, ret_evt
        )
        @return_event ret_evt[]
    end
end

