# memory operations

## managed memory

# to safely use allocated memory across tasks and devices, we don't simply return raw
# memory objects, but wrap them in a manager that ensures synchronization and ownership.

# XXX: immutable with atomic refs?
mutable struct Managed{M}
    const mem::M

    # which stream is currently using the memory.
    queue::cl.CmdQueue

    # whether there are outstanding operations that haven't been synchronized
    dirty::Bool

    # whether the memory has been captured in a way that would make the dirty bit unreliable
    captured::Bool

    function Managed(mem::cl.AbstractBuffer; queue = cl.queue(), dirty = true, captured = false)
        # NOTE: memory starts as dirty, because stream-ordered allocations are only
        #       guaranteed to be physically allocated at a synchronization event.
        return new{typeof(mem)}(mem, queue, dirty, captured)
    end
end

# wait for the current owner of memory to finish processing
function synchronize(managed::Managed)
    cl.finish(managed.queue)
    return managed.dirty = false
end

function maybe_synchronize(managed::Managed)
    return if managed.dirty || managed.captured
        synchronize(managed)
    end
end

function Base.convert(::Type{CLPtr{T}}, managed::Managed{M}) where {T, M}
    # let null pointers pass through as-is
    ptr = convert(CLPtr{T}, managed.mem)
    if ptr == cl.CL_NULL
        return ptr
    end

    #= TODO: FIGURE OUT ACTIVE STATE
  # state = cl.active_state()

  # accessing memory on another device: ensure the data is ready and accessible
  if M == cl.DeviceBuffer && state.context != managed.mem.ctx
    maybe_synchronize(managed)
    # source_device = managed.mem.dev

    # TODO: Look into P2P access for OpenCL
    #=
    # enable peer-to-peer access
    if maybe_enable_peer_access(state.device, source_device) != 1
        throw(ArgumentError(
            """cannot take the GPU address of inaccessible device memory.

               You are trying to use memory from GPU $(deviceid(source_device)) on GPU $(deviceid(state.device)).
               P2P access between these devices is not possible; either switch to GPU $(deviceid(source_device))
               by calling `CUDA.device!($(deviceid(source_device)))`, or copy the data to an array allocated on device $(deviceid(state.device))."""))
    end

    # set pool visibility
    if stream_ordered(source_device)
      pool = pool_create(source_device)
      access!(pool, state.device, ACCESS_FLAGS_PROT_READWRITE)
    end
    =#
  end

  # accessing memory on another stream: ensure the data is ready and take ownership
  if managed.queue != state.stream
    maybe_synchronize(managed)
    managed.queue = state.stream
  end
  =#

    managed.dirty = true
    return ptr
end

function Base.convert(::Type{Ptr{T}}, managed::Managed{M}) where {T, M}
    # let null pointers pass through as-is
    ptr = convert(Ptr{T}, managed.mem)
    if ptr == C_NULL
        return ptr
    end

    # accessing memory on the CPU: only allowed for host or unified allocations
    if M == cl.DeviceBuffer
        throw(
            ArgumentError(
                """cannot take the CPU address of GPU memory."""
            )
        )

    end

    # make sure any work on the memory has finished.
    maybe_synchronize(managed)
    return ptr
end
#=
function Base.unsafe_copyto!(dst::Ptr, src::Ptr, nbytes::Integer; blocking::Bool=false,
                            wait_for::Vector{Event}=Event[])
    n_evts  = length(wait_for)
    evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        ret_evt = Ref{cl_event}()
        clEnqueueSVMMemcpy(queue(), blocking, dst, src, nbytes, n_evts, evt_ids, ret_evt)
        @return_event ret_evt[]
    end
end


function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device, dst::Union{Ptr{T},CLPtr{T}},
                             src::Union{Ptr{T},CLPtr{T}}, N::Integer) where T
    bytes = N*sizeof(T)
    bytes==0 && return
    
    
    execute!(global_queue(ctx, dev)) do list
        append_copy!(list, dst, src, bytes)
    end
end


function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device, dst::Union{Ptr{T},cl.CLPtr{T}},
                             src::Union{Ptr{T},cl.CLPtr{T}}, N::Integer, queu::cl.CmdQueue=cl.queue(), blocking=false, wait_event_list::Vector{cl.Event}=cl.Event[]) where T
    bytes = N*sizeof(T)
    bytes == 0 && return

    evt_ids = isempty(wait_event_list) ? C_NULL : [pointer(evt) for evt in wait_event_list]
    
    cl.ext_clEnqueueMemcpyINTEL(
            queu,
            blocking,
            reinterpret(Ptr{Nothing}, dst),
            reinterpret(Ptr{Nothing}, src),
            bytes,
            length(wait_event_list),
            evt_ids,
            Ref{cl.cl_event}()
      )
    cl.finish(queu)
    return dst
end
#
function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device, dst::Union{Ptr{T},cl.CLPtr{T}},
                             src::Union{Ptr{T},cl.CLPtr{T}}, N::Integer) where T
    bytes = N*sizeof(T)
    bytes==0 && return
    execute!(global_queue(ctx, dev)) do list
        append_copy!(list, dst, src, bytes)
    end
end

function unsafe_fill!(ctx::cl.Context, dev::cl.Device, ptr::Union{Ptr{T},cl.CLPtr{T}},
                      pattern::Union{Ptr{T},cl.CLPtr{T}}, N::Integer) where T
    bytes = N*sizeof(T)
    bytes==0 && return
    execute!(global_queue(ctx, dev)) do list
        append_fill!(list, ptr, pattern, sizeof(T), bytes)
    end
end
=#

function Base.unsafe_copyto!(
        ::cl.Context, ::cl.Device, dst::Union{CLPtr{T}, Ptr{T}}, src::Union{CLPtr{T}, Ptr{T}}, N::Integer;
        queu::cl.CmdQueue = cl.queue()
    ) where {T}

    cl.enqueue_usm_memcpy(dst, src, N * sizeof(T); queu = queu)

    cl.finish(queu)
    return dst
end

function unsafe_fill!(
        ctx::cl.Context, dev::cl.Device, ptr::Union{Ptr{T}, CLPtr{T}},
        pattern::Union{Ptr{T}, CLPtr{T}}, N::Integer; queu::cl.CmdQueue = cl.queue()
    ) where {T}
    bytes = N * sizeof(T)
    bytes == 0 && return
    cl.enqueue_usm_memfill(ptr, pattern, sizeof(T), bytes; queu = queu)
    return cl.finish(queu)
end
