# --- low level OpenCL Event ---
abstract CLEvent

type Event <: CLEvent
    id :: CL_event

    function Event(evt_id::CL_event; retain=false)
        if retain
            @check api.clRetainEvent(evt_id)
        end
        evt = new(evt_id)
        finalizer(evt, evt -> release!(evt))
        return evt
    end
end

# wait for completion before running finalizer
type NannyEvent <: CLEvent
    id::CL_event
    obj::Any

    function NannyEvent(evt_id::CL_event, obj::Any; retain=false)
        if retain
            @check api.clRetainEvent(evt_id)
        end
        nanny_evt = new(evt_id, obj)
        finalizer(nanny_evt, x -> begin 
            wait(x)
            x.obj = nothing
            release!(x)
        end)
        nanny_evt
    end
end

NannyEvent(evt::Event, obj::Any; retain=false) = NannyEvent(evt.id, obj, retain=retain)

function release!(evt::CLEvent)
    if evt.id != C_NULL
        @check api.clReleaseEvent(evt.id)
        evt.id = C_NULL
    end
end

Base.pointer(evt::CLEvent) = evt.id
@ocl_object_equality(CLEvent)

function Base.show(io::IO, evt::Event)
    ptr_address = "0x$(hex(unsigned(Base.pointer(evt)), WORD_SIZE>>2))"
    print(io, "OpenCL.Event(@$ptr_address)")
end

Base.getindex(evt::CLEvent, evt_info::Symbol) = info(evt, evt_info)

@ocl_v1_1_only begin

    type UserEvent <: CLEvent
        id :: CL_event 

        function UserEvent(evt_id::CL_event; retain=true)
            if retain
                @check api.clRetainEvent(evt_id)
            end
            evt = new(evt_id)
            finalizer(evt, x -> release!(x))
            return evt
        end
    end
    
    function UserEvent(ctx::Context)
        status = Array(CL_int, 1)
        evt_id = api.clCreateUserEvent(ctx.id, status)
        if status[1] != CL_SUCCESS
            throw(CLError(status[1]))
        end
        try
            return UserEvent(evt_id, retain=false)
        catch err
            @check api.clReleaseEvent(evt_id)
            throw(err)
        end
    end

    function Base.show(io::IO, evt::UserEvent)
        ptr_address = "0x$(hex(unsigned(Base.pointer(evt)), WORD_SIZE>>2))"
        print(io, "OpenCL.UserEvent(@$ptr_address)")
    end

    function complete(evt::UserEvent)
        @check api.clSetUserEventStatus(evt.id, CL_COMPLETE)
        return evt
    end
end

function event_notify(evt_id::CL_event, status::CL_int, julia_func::Ptr{Void})
    callback = unsafe_pointer_to_objref(julia_func)::Function
    callback(evt_id, status)
    return C_NULL::Ptr{Void}
end

const event_notify_ptr = cfunction(event_notify, Ptr{Void},
                                   (CL_event, CL_int, Ptr{Void}))

function add_callback(evt::CLEvent, callback::Function)
    @check api.clSetEventCallback(evt.id, CL_COMPLETE, event_notify_ptr, callback)
end

function wait(evt::CLEvent)
    evt_id = [evt.id]
    @check api.clWaitForEvents(cl_uint(1), evt_id)
    return evt
end

function wait(evts::Vector{CLEvent})
    evt_ids = [evt.id for evt in evts]
    if !isempty(evt_ids)
        nevents = cl_uint(length(evt_ids))
        @check api.clWaitForEvents(nevents, evt_ids)
    end
    return evts
end

@ocl_v1_2_only begin
    function enqueue_marker_with_wait_list(q::CmdQueue,
                                           wait_for::Vector{CLEvent})
        n_wait_events = cl_uint(length(wait_for))
        wait_evt_ids = [evt.id for evt in wait_for]
        ret_evt = Array(CL_event, 1)
        @check api.clEnqueueMarkerWithWaitList(q.id, n_wait_events,
                                               isempty(wait_evt_ids)? C_NULL : wait_evt_ids,
                                               ret_evt)
        @return_event ret_evt[1]
    end

    function enqueue_barrier_with_wait_list(q::CmdQueue,
                                            wait_for::Vector{CLEvent})
        n_wait_events = cl_uint(length(wait_for))
        wait_evt_ids = [evt.id for evt in wait_for]
        ret_evt = Array(CL_event, 1)
        @check api.clEnqueueBarrierWithWaitList(q.id, n_wait_events,
                                                isempty(wait_evt_ids)? C_NULL : wait_evt_ids,
                                                ret_evt)
        @return_event ret_evt[1]
    end
end

function enqueue_marker(q::CmdQueue)
    evt = Array(CL_event, 1)
    @check api.clEnqueueMarker(q.id, evt)
    @return_event evt[1]
end
@deprecate enqueue_marker enqueue_marker_with_wait_list

function enqueue_wait_for_events{T<:CLEvent}(q::CmdQueue, wait_for::Vector{T})
    n_wait_events = cl_uint(length(wait_for))
    wait_evt_ids = [evt.id for evt in wait_for]
    @check api.clEnqueueWaitForEvents(q.id, n_wait_events,
                                      isempty(wait_evt_ids) ? C_NULL : wait_evt_ids)
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
            time = CL_long[0]
            err_code = api.clGetEventProfilingInfo(evt.id, $profile_info,
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
            return time[1]
        end
    end
end
    

let command_queue(evt::CLEvent) = begin
        cmd_q = Array(CL_command_queue, 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_QUEUE,
                                  sizeof(CL_command_queue), cmd_q, C_NULL)
        return CmdQueue(cmd_q[1])
    end
    
    command_type(evt::CLEvent) = begin
        cmd_t = Array(CL_int , 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_TYPE,
                                  sizeof(CL_int), cmd_t, C_NULL)
        return cmd_t[1]
    end

    reference_count(evt::CLEvent) = begin
        cnt = Array(CL_uint, 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_REFERENCE_COUNT,
                                  sizeof(CL_uint), cnt, C_NULL)
        return cnt[1]
    end

    context(evt::CLEvent) = begin
        ctx = Array(CL_context, 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_CONTEXT,
                                  sizeof(CL_context), CL_context, C_NULL)
        Context(ctx[1])
    end

    status(evt::CLEvent) = begin
        st = Array(CL_int, 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                  sizeof(CL_int), st, C_NULL)
        status = st[1]
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

    const info_map = (Symbol => Function)[
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
    ]

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
