# OpenCL.Event

abstract type AbstractEvent <: CLObject end

mutable struct Event <: AbstractEvent
    const id::cl_event

    function Event(evt_id; retain::Bool=false)
        evt = new(evt_id)
        retain && clRetainEvent(evt)
        finalizer(clReleaseEvent, evt)
        return evt
    end
end

# wait for completion before running finalizer
mutable struct NannyEvent <: AbstractEvent
    const id::cl_event
    const obj::Any

    function NannyEvent(evt_id, obj; retain::Bool=false)
        nanny_evt = new(evt_id, obj)
        retain && clRetainEvent(nanny_evt)
        finalizer(clReleaseEvent, nanny_evt)
        nanny_evt
    end
end

NannyEvent(evt::Event, obj; retain=false) = NannyEvent(evt.id, obj; retain)

macro return_event(evt)
    quote
        evt = $(esc(evt))
        try
            return Event(evt, retain=false)
        catch err
            clReleaseEvent(evt)
            throw(err)
        end
    end
end

macro return_nanny_event(evt, obj)
    quote
        evt = $(esc(evt))
        try
            return NannyEvent(evt, $(esc(obj)))
        catch err
            clReleaseEvent(evt)
            throw(err)
        end
    end
end

Base.unsafe_convert(::Type{cl_event}, evt::AbstractEvent) = evt.id

function Base.show(io::IO, evt::Event)
    ptr_val = convert(UInt, pointer(evt))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Event(@$ptr_address)")
end

mutable struct UserEvent <: AbstractEvent
    const id::cl_event

    function UserEvent(evt_id::cl_event, retain::Bool=false)
        evt = new(evt_id)
        retain && clRetainEvent(evt)
        finalizer(clReleaseEvent, evt)
        return evt
    end
end

function UserEvent(; retain=false)
    status = Ref{Cint}()
    evt_id = clCreateUserEvent(context(), status)
    if status[] != CL_SUCCESS
        throw(CLError(status[]))
    end
    try
        return UserEvent(evt_id, retain)
    catch err
        clReleaseEvent(evt_id)
        throw(err)
    end
end

function Base.show(io::IO, evt::UserEvent)
    ptr_val = convert(UInt, pointer(evt))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.UserEvent(@$ptr_address)")
end

function complete(evt::UserEvent)
    clSetUserEventStatus(evt, CL_COMPLETE)
    return evt
end

struct _EventCB
    handle::Ptr{Nothing}
    evt_id::cl_event
    status::Cint
end

function event_notify(evt_id::cl_event, status::Cint, payload::Ptr{Nothing})
    ptr = convert(Ptr{_EventCB}, payload)
    handle = unsafe_load(ptr).handle

    val = _EventCB(handle, evt_id, status)
    unsafe_store!(ptr, val)

    # Use uv_async_send to notify the main thread
    ccall(:uv_async_send, Nothing, (Ptr{Nothing},), handle)
    nothing
end

function add_callback(evt::AbstractEvent, callback::Function)
    event_notify_ptr = @cfunction(event_notify, Nothing,
                                  (cl_event, Cint, Ptr{Cvoid}))

    # The uv_callback is going to notify a task that,
    # then executes the real callback.
    cb = Base.AsyncCondition()
    GC.@preserve cb begin

        # Storing the results of our c_callback needs to be
        # isbits && isimmutable
        r_ecb = Ref(_EventCB(Base.unsafe_convert(Ptr{Cvoid}, cb), 0, 0))

        clSetEventCallback(evt, CL_COMPLETE, event_notify_ptr, r_ecb)

        @async begin
           try
             Base.wait(cb)
             ecb = r_ecb[]
             callback(ecb.evt_id, ecb.status)
           catch
             rethrow()
           finally
             Base.close(cb)
           end
        end
    end
end

function Base.wait(evt::AbstractEvent)
    evt_id = [evt.id]
    clWaitForEvents(cl_uint(1), evt_id)
    return evt
end

function Base.wait(evts::Vector{AbstractEvent})
    isempty(evts) && return evts
    evt_ids = [pointer(evt) for evt in evts]
    GC.@preserve evts begin
        clWaitForEvents(ength(evt_ids), evt_ids)
    end
    return evts
end

function enqueue_marker_with_wait_list(wait_for::Vector{AbstractEvent})
    n_wait_events = cl_uint(length(wait_for))
    wait_evt_ids = [evt.id for evt in wait_for]
    ret_evt = Ref{cl_event}()
    clEnqueueMarkerWithWaitList(queue(), n_wait_events,
                                isempty(wait_evt_ids) ? C_NULL : wait_evt_ids,
                                ret_evt)
    @return_event ret_evt[]
end

function enqueue_barrier_with_wait_list(wait_for::Vector{AbstractEvent})
    n_wait_events = cl_uint(length(wait_for))
    wait_evt_ids = [evt.id for evt in wait_for]
    ret_evt = Ref{cl_event}()
    clEnqueueBarrierWithWaitList(queue(), n_wait_events,
                                 isempty(wait_evt_ids) ? C_NULL : wait_evt_ids,
                                 ret_evt)
    @return_event ret_evt[]
end

function enqueue_marker()
    evt = Ref{cl_event}()
    clEnqueueMarker(queue(), evt)
    @return_event evt[]
end
@deprecate enqueue_marker enqueue_marker_with_wait_list

function enqueue_wait_for_events(wait_for::Vector{T}) where {T<:AbstractEvent}
    wait_evt_ids = isempty(wait_for) ? C_NULL : [pointer(evt) for evt in wait_for]
    GC.@preserve wait_for begin
        clEnqueueWaitForEvents(queue(), length(wait_for), wait_evt_ids)
   end
end

function enqueue_wait_for_events(wait_for::AbstractEvent)
    enqueue_wait_for_events([wait_for])
end

function enqueue_barrier()
    clEnqueueBarrier(queue())
    return
end
@deprecate enqueue_barrier enqueue_barrier_with_wait_list

cl_event_status(s::Symbol) = begin
    if s == :queued
        return CL_QUEUED
    elseif s == :submitted
        return CL_SUBMITTED
    elseif s == :running
        return CL_RUNNING
    elseif s == :complete
        return CL_COMPLETE
    else
        throw(ArgumentError("unrecognized status symbol :$s"))
    end
end

function Base.getproperty(evt::AbstractEvent, s::Symbol)
    function profiling_info(evt::AbstractEvent, profile_info)
        time = Ref{Clong}(0)
        try
            clGetEventProfilingInfo(evt, profile_info, sizeof(cl_ulong), time, C_NULL)
        catch err
            if isa(err, CLError) && err.code == CL_PROFILING_INFO_NOT_AVAILABLE
                if evt.status != :complete
                    throw(ArgumentError("Event is not complete; cannot access profiling info yet"))
                else
                    throw(ArgumentError("Command queue does not support profiling; consider running under `cl.queue!(:profiling)`"))
                end
            end
            rethrow()
        end
        return time[]
    end

    # regular properties
    if s == :context
        ctx = Ref{cl_context}()
        clGetEventInfo(evt, CL_EVENT_CONTEXT, sizeof(cl_context), ctx, C_NULL)
        return Context(ctx[])
    elseif s == :command_queue
        cmd_q = Ref{cl_command_queue}()
        clGetEventInfo(evt, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), cmd_q, C_NULL)
        return CmdQueue(cmd_q[])
    elseif s == :command_type
        cmd_t = Ref{Cint}()
        clGetEventInfo(evt, CL_EVENT_COMMAND_TYPE, sizeof(Cint), cmd_t, C_NULL)
        return cmd_t[]
    elseif s == :reference_count
        cnt = Ref{Cuint}()
        clGetEventInfo(evt, CL_EVENT_REFERENCE_COUNT, sizeof(Cuint), cnt, C_NULL)
        return Int(cnt[])
    elseif s == :status
        st = Ref{Cint}()
        clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(Cint), st, C_NULL)
        status = st[]
        if status == CL_QUEUED
            return :queued
        elseif status == CL_SUBMITTED
            return :submitted
        elseif status == CL_RUNNING
            return :running
        elseif status == CL_COMPLETE
            return :complete
        else
            throw(ArgumentError("Unknown status value: $status"))
        end

    # profiling properties
    elseif s == :profile_start
        return profiling_info(evt, CL_PROFILING_COMMAND_START)
    elseif s == :profile_end
        return profiling_info(evt, CL_PROFILING_COMMAND_END)
    elseif s == :profile_queued
        return profiling_info(evt, CL_PROFILING_COMMAND_QUEUED)
    elseif s == :profile_submit
        return profiling_info(evt, CL_PROFILING_COMMAND_SUBMIT)
    elseif s == :profile_duration
        return evt.profile_end - evt.profile_start

    else
        return getfield(evt, s)
    end
end
