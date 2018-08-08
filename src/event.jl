# OpenCL.Event

abstract type CLEvent <: CLObject end

mutable struct Event <: CLEvent
    id :: CL_event

    function Event(evt_id::CL_event; retain=false)
        if retain
            @check api.clRetainEvent(evt_id)
        end
        evt = new(evt_id)
        finalizer(_finalize, evt)
        return evt
    end
end

# wait for completion before running finalizer
mutable struct NannyEvent <: CLEvent
    id::CL_event
    obj::Any

    function NannyEvent(evt_id::CL_event, obj::Any; retain=false)
        if retain
            @check api.clRetainEvent(evt_id)
        end
        nanny_evt = new(evt_id, obj)
        finalizer(x -> begin
            x.id != C_NULL && wait(x)
            x.obj = nothing
            _finalize(x)
        end, nanny_evt)
        nanny_evt
    end
end

function _finalize(evt::CLEvent)
    if evt.id != C_NULL
        @check api.clReleaseEvent(evt.id)
        evt.id = C_NULL
    end
end

NannyEvent(evt::Event, obj::Any; retain=false) = NannyEvent(evt.id, obj, retain=retain)

Base.pointer(evt::CLEvent) = evt.id

function Base.show(io::IO, evt::Event)
    ptr_val = convert(UInt, Base.pointer(evt))
    ptr_address = "0x$(hex(ptr_val, Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Event(@$ptr_address)")
end

Base.getindex(evt::CLEvent, evt_info::Symbol) = info(evt, evt_info)

@ocl_v1_1_only begin

    mutable struct UserEvent <: CLEvent
        id :: CL_event

        function UserEvent(evt_id::CL_event, retain=false)
            if retain
                @check api.clRetainEvent(evt_id)
            end
            evt = new(evt_id)
            finalizer(_finalize, evt)
            return evt
        end
    end

    function UserEvent(ctx::Context; retain=false)
        status = Ref{CL_int}()
        evt_id = api.clCreateUserEvent(ctx.id, status)
        if status[] != CL_SUCCESS
            throw(CLError(status[]))
        end
        try
            return UserEvent(evt_id, retain)
        catch err
            @check api.clReleaseEvent(evt_id)
            throw(err)
        end
    end

    function Base.show(io::IO, evt::UserEvent)
        ptr_val = convert(UInt, Base.pointer(evt))
        ptr_address = "0x$(hex(ptr_val, Sys.WORD_SIZE>>2))"
        print(io, "OpenCL.UserEvent(@$ptr_address)")
    end

    function complete(evt::UserEvent)
        @check api.clSetUserEventStatus(evt.id, CL_COMPLETE)
        return evt
    end
end

struct _EventCB
    handle :: Ptr{Cvoid}
    evt_id :: CL_event
    status :: CL_int
end

function event_notify(evt_id::CL_event, status::CL_int, payload::Ptr{Cvoid})
    ptr = convert(Ptr{_EventCB}, payload)
    handle = unsafe_load(ptr, 1).handle

    val = _EventCB(handle, evt_id, status)
    unsafe_store!(ptr, val, 1)

    # Use uv_async_send to notify the main thread
    ccall(:uv_async_send, Cvoid, (Ptr{Cvoid},), handle)
    nothing
end

function add_callback(evt::CLEvent, callback::Function)
    event_notify_ptr = @cfunction(event_notify, Nothing,
                                  (CL_event, CL_int, Ptr{Cvoid}))

    # The uv_callback is going to notify a task that,
    # then executes the real callback.
    cb = Base.AsyncCondition()

    # Storing the results of our c_callback needs to be
    # isbits && isimmutable
    r_ecb = Ref(_EventCB(Base.unsafe_convert(Ptr{Cvoid}, cb), 0, 0))

    @check api.clSetEventCallback(evt.id, CL_COMPLETE, event_notify_ptr, r_ecb)

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

function wait(evt::CLEvent)
    evt_id = [evt.id]
    @check api.clWaitForEvents(cl_uint(1), pointer(evt_id))
    return evt
end

function wait(evts::Vector{CLEvent})
    evt_ids = [evt.id for evt in evts]
    if !isempty(evt_ids)
        nevents = cl_uint(length(evt_ids))
        @check api.clWaitForEvents(nevents, pointer(evt_ids))
    end
    return evts
end

@ocl_v1_2_only begin
    function enqueue_marker_with_wait_list(q::CmdQueue,
                                           wait_for::Vector{CLEvent})
        n_wait_events = cl_uint(length(wait_for))
        wait_evt_ids = [evt.id for evt in wait_for]
        ret_evt = Ref{CL_event}()
        @check api.clEnqueueMarkerWithWaitList(q.id, n_wait_events,
                                               isempty(wait_evt_ids) ? C_NULL : wait_evt_ids,
                                               ret_evt)
        @return_event ret_evt[]
    end

    function enqueue_barrier_with_wait_list(q::CmdQueue,
                                            wait_for::Vector{CLEvent})
        n_wait_events = cl_uint(length(wait_for))
        wait_evt_ids = [evt.id for evt in wait_for]
        ret_evt = Ref{CL_event}()
        @check api.clEnqueueBarrierWithWaitList(q.id, n_wait_events,
                                                isempty(wait_evt_ids) ? C_NULL : wait_evt_ids,
                                                ret_evt)
        @return_event ret_evt[]
    end
end

function enqueue_marker(q::CmdQueue)
    evt = Ref{CL_event}()
    @check api.clEnqueueMarker(q.id, evt)
    @return_event evt[]
end
@deprecate enqueue_marker enqueue_marker_with_wait_list

function enqueue_wait_for_events(q::CmdQueue, wait_for::Vector{T}) where T<:CLEvent
    n_wait_events = cl_uint(length(wait_for))
    wait_evt_ids = [evt.id for evt in wait_for]
    @check api.clEnqueueWaitForEvents(q.id, n_wait_events,
                                      isempty(wait_evt_ids) ? C_NULL : pointer(wait_evt_ids))
end

function enqueue_wait_for_events(q::CmdQueue, wait_for::CLEvent)
    enqueue_wait_for_events(q, [wait_for])
end

function enqueue_barrier(q::CmdQueue)
    @check api.clEnqueueBarrier(q.id)
    return q
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

macro profile_info(func, profile_info)
    quote
        function $(esc(func))(evt::CLEvent)
            time = Ref{CL_long}(0)
            err_code = api.clGetEventProfilingInfo(evt.id, $(esc(profile_info)),
                                                   sizeof(CL_ulong), time, C_NULL)
            if err_code != CL_SUCCESS
                if err_code == CL_PROFILING_INFO_NOT_AVAILABLE
                    if evt[:status] != :complete
                        #TODO: evt must have status complete before it can be profiled
                        throw(CLError(err_code))
                    else
                        #TODO: queue must be created with :profile argument
                        throw(CLError(err_code))
                    end
                end
                throw(CLError(err_code))
            end
            return time[]
        end
    end
end


let command_queue(evt::CLEvent) = begin
        cmd_q = Ref{CL_command_queue}()
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_QUEUE,
                                  sizeof(CL_command_queue), cmd_q, C_NULL)
        return CmdQueue(cmd_q[])
    end

    command_type(evt::CLEvent) = begin
        cmd_t = Ref{CL_int}()
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_TYPE,
                                  sizeof(CL_int), cmd_t, C_NULL)
        return cmd_t[]
    end

    reference_count(evt::CLEvent) = begin
        cnt = Ref{CL_uint}()
        @check api.clGetEventInfo(evt.id, CL_EVENT_REFERENCE_COUNT,
                                  sizeof(CL_uint), cnt, C_NULL)
        return cnt[]
    end

    context(evt::CLEvent) = begin
        ctx = Ref{CL_context}()
        @check api.clGetEventInfo(evt.id, CL_EVENT_CONTEXT,
                                  sizeof(CL_context), CL_context, C_NULL)
        Context(ctx[])
    end

    status(evt::CLEvent) = begin
        st = Ref{CL_int}()
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                  sizeof(CL_int), st, C_NULL)
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
    end

    @profile_info(profile_start,  CL_PROFILING_COMMAND_START)
    @profile_info(profile_end,    CL_PROFILING_COMMAND_END)
    @profile_info(profile_queued, CL_PROFILING_COMMAND_QUEUED)
    @profile_info(profile_submit, CL_PROFILING_COMMAND_SUBMIT)

    profile_duration(evt::Event) = begin
        return evt[:profile_end] - evt[:profile_start]
    end

    info_map = Dict{Symbol, Function}(
        :context => context,
        :command_queue => command_queue,
        :reference_count => reference_count,
        :command_type => command_type,
        :status => status,
        :profile_start => profile_start,
        :profile_end => profile_end,
        :profile_queued => profile_queued,
        :profile_submit => profile_submit,
        :profile_duration => profile_duration,
    )

    function info(evt::CLEvent, evt_info::Symbol)
        try
            func = info_map[evt_info]
            func(evt)
        catch err
            if isa(err, KeyError)
                throw(ArgumentError("OpenCL.Event has no info for: $evt_info"))
            else
                throw(err)
            end
        end
    end
end
