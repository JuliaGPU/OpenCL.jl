# --- low level OpenCL Event ---

abstract CLEvent

type Event <: CLEvent
    id :: CL_event

    function Event(evt_id::CL_event; retain=true)
        if retain
            @check api.clRetainEvent(evt_id)
        end
        evt = new(evt_id)
        finalizer(evt, evt -> release!(evt))
        return evt
    end
end

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
    print(io, "<OpenCL.Event @$ptr_address>")
end

@ocl_v1_1_only begin

    type UserEvent <: CLEvent
        id :: CL_event 

        function UserEvent(evt_id::CL_event; retain=true)
            if retain
                @check api.clRetainEvent(evt_id)
            end
            evt = new(evt_id)
            finalizer(evt, evt -> release!(evt))
            return evt
        end
    end
    
    function Base.show(io::IO, evt::UserEvent)
        ptr_address = "0x$(hex(unsigned(Base.pointer(evt)), WORD_SIZE>>2))"
        print(io, "<OpenCL.UserEvent @$ptr_address>")
    end

    function set_status(evt::UserEvent, 
                       exec_status::CL_int)
        @check api.clSetUserEventStatus(evt.id, exec_status)
    end
end


function wait(evt::CLEvent)
    evt_id = [evt.id]
    @check api.clWaitForEvents(1, evt_id)
    return evt
end

function wait(evts::Vector{CLEvent})
    evt_ids = [evt.id for evt in evts]
    if !isempty(evt_ids)
        @check api.clWaitForEvents(length(evt_ids), evt_ids)
    end
    return evts
end

@ocl_v1_2_only begin
    function enqueue_marker_with_wait_list(q::CommandQueue,
                                           wait_for::Vector{CLEvent})
        n_wait_events = cl_uint(length(wait_for))
        wait_evt_ids = [evt.id for evt in wait_for]
        ret_evt = Array(CL_event, 1)
        @check api.clEnqueueMarkerWithWaitList(q.id, n_wait_events,
                            isempty(wait_evt_ids) ? C_NULL : wait_evt_ids, ret_evt)
        @return_event ret_evt[1]
    end

    function enqueue_barrier_with_wait_list(q::CommandQueue,
                                            wait_for::Vector{CLEvent})
        n_wait_events = cl_uint(length(wait_for))
        wait_evt_ids = [evt.id for evt in wait_for]
        ret_evt = Array(CL_event, 1)
        @check api.clEnqueueBarrierWithWaitList(q.id, n_wait_events,
                            isempty(wait_evt_ids) ? C_NULL : wait_evt_ids, ret_evt)
        @return_event ret_evt[1]
    end
end

# internal (pre 1.2 contexts)
function enqueue_marker(q::CommandQueue)
    evt = Array(CL_event, 1)
    @check api.clEnqueueMarker(q.id, evt)
    @return_event evt[1]
end

function enqueue_wait_for_events(q::CommandQueue, wait_for::Vector{CLEvent})
    n_wait_events = cl_uint(length(wait_for))
    wait_evt_ids = [evt.id for evt in wait_for]
    ret_evt = Array(CL_event, 1)
    @check api.clEnqueueWaitForEvents(q.id, n_wait_events, 
                           isempty(wait_evt_ids) ? C_NULL : wait_evt_ids, ret_evt)
    @return_event ret_evt[1]
end

function enqueue_barrier(q::CommandQueue)
    @check api.clEnqueueBarrier(q.id)
    return q
end

#TODO: make the following more consistent
let status_dict = (CL_uint => Symbol)[
                   CL_QUEUED => :queued,
                   CL_SUBMITTED => :submitted,
                   CL_RUNNING => :running,
                   CL_COMPLETE => :complete]

    function status(evt::CLEvent)
        status = Array(CL_uint, 1)
        @check api.clGetEventProfilingInfo(evt.id, 
                                           CL_EVENT_COMMAND_EXECUTION_STATUS, 
                                           sizeof(CL_uint), status)
        return status_dict[status[1]]
    end
end

function status(evt_id::CL_event)
    status = Array(CL_int, 1)
    @check api.clGetEventInfo(evt_id, CL_EVENT_COMMAND_EXECUTION_STATUS,
                              sizeof(CL_int), status)
    return status[1] 
end

function profiling_info(evt::CLEvent, param::CL_profiling_info)
    if     param == CL_PROFILING_COMMAND_QUEUED
    elseif param == CL_PROFILING_COMMAND_SUBMIT
    elseif param == CL_PROFILING_COMMAND_START
    elseif param == CL_PROFILING_COMMAND_END
        len = Array(CL_ulong, 1)
        @check api.clGetEventProfilingInfo(evt.id, param, sizeof(CL_ulong),
                                           len, C_NULL)
        return len[1] 
    else
        throw(CLError(CL_INVALID_VALUE))
    end
end

# cannot use reserved word end as symbol
function profiling_info(evt::CLEvent, param::Symbol)
    if     param == :pqueued
    elseif param == :psubmitted
    elseif param == :pstart
    elseif param == :pend 
        return profiling_info(evt, CL_PROFILING_COMMAND_END)
    else
        throw(CLError(CL_INVALID_VALUE))
    end
end


let command_queue(evt::CLEvent) = begin
        cmd_q = Array(CL_command_queue, 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_QUEUE,
                                  sizeof(CL_command_queue), cmd_q, C_NULL)
        return CommandQueue(cmd_q[1])
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
    
    const info_map = (Symbol => Function)[
        :context => context,
        :command_queue => command_queue,
        :reference_count => reference_count,
        :command_type => command_type,
        :status => status,
    ]

    function info(q::CLEvent, qinfo::Symbol)
        try
            func = info_map[qinfo]
            func(d)
        catch err
            if isa(err, KeyError)
                error("OpenCL.Event has no info for: $qinfo") 
            else
                throw(err)
            end
        end
    end
end

#function clGetEventProfilingInfo(evt_id::CL_event,
#                                 param::CL_profiling_info,
#                                 param_size::Csize_t,
#                                 param_val::Ptr{Void},
#                                 param_value_size_ret::Ptr{Csize_t})
#    err = ccall((:clGetEventProfilingInfo, libopencl),
#                CL_int,
#                (CL_event, CL_profiling_info, Csize_t, Ptr{Void}, Ptr{Csize_t}),
#                evt_id, param, param_size, param_val, param_value_size_ret)
#    if err != CL_SUCCESS
#        if err == CL_PROFILING_INFO_NOT_AVAILABLE
#            if status(evt_id) != CL_COMPLETE
#                #TODO: throw
#                error("Event must have :completed status before it can be profiled")
#            else
#                error("Queue must be created with profile=True")
#            end
#        error("OpenCL clGetEventProfilingInfo error...")
#    end
#    return status
#end
#
#function profile_start(evt::Event)
#    status = convert(Ptr{Void}, Array(CL_ulong, 1))
#    clGetEventProfilingInfo(evt.id, CL_PROFILING_COMMAND_START, sizeof(CL_ulong), status, C_NULL)
#    return unsafe_ref(convert(Ptr{CL_ulong}, status))
#end
#
#function profile_end(evt::Event)
#    status = convert(Ptr{Void}, Array(CL_ulong, 1))
#    clGetEventProfilingInfo(evt.id, CL_PROFILING_COMMAND_END, sizeof(CL_ulong), status, C_NULL)
#    return unsafe_ref(convert(Ptr{CL_ulong}, status))
#end
#
#function duration(evt::Event)
#    profile_end(evt) - profile_start(evt)
#end
#
#function profile_queued(evt::Event)
#    status = convert(Ptr{Void}, Array(CL_ulong, 1))
#    clGetEventProfilingInfo(evt.id, CL_PROFILING_COMMAND_QUEUED, sizeof(CL_ulong), status, C_NULL)
#    return unsafe_ref(convert(Ptr{CL_ulong}, status))
#end
#
#function profile_submitted(evt::Event)
#    status = convert(Ptr{Void}, Array(CL_ulong, 1))
#    clGetEventProfilingInfo(evt.id, CL_PROFILING_COMMAND_SUBMIT, sizeof(CL_ulong), status, C_NULL)
#    return unsafe_ref(convert(Ptr{CL_ulong}, status))
#end
