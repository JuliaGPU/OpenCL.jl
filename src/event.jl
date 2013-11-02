# --- low level OpenCL Event ---

type Event
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

function release!(evt::Event)
    if evt.id != C_NULL
        @check api.clReleaseEvent(evt_id)
        evt.id = C_NULL
    end
end

Base.pointer(evt::Event) = evt.id
@ocl_object_equality(Event)

function Base.show(io::IO, q::CommandQueue)
    ptr_address = "0x$(hex(unsigned(Base.pointer(q)), WORD_SIZE>>2))"
    print(io, "<OpenCL.Event @$ptr_address>")
end


function wait(evt::Event)
    evt_id = [evt.id]
    @check api.clWaitForEvents(1, evt_id)
    return evt
end

#TODO: wait for multiple events by passing in array
function wait(evts::Vector{Event})
    for evt in evts
        wait(evt)
    end
    return evts
end

#TODO: make the following more consistent
let status_dict = (CL_uint => Symbol)[
                   CL_QUEUED => :queued,
                   CL_SUBMITTED => :submitted,
                   CL_RUNNING => :running,
                   CL_COMPLETE => :complete]

    function status(evt::Event)
        status = Array(CL_uint, 1)
        @check api.clGetEventProfilingInfo(evt.id, 
                                           CL_EVENT_COMMAND_EXECUTION_STATUS, 
                                           sizeof(CL_uint), status)
        return status_dict[status[1]]
    end
end

function status(evt_id::CL_event)
    status = convert(Ptr{Void}, Array(CL_int, 1))
    @check api.clGetEventInfo(evt_id, CL_EVENT_COMMAND_EXECUTION_STATUS,
                              sizeof(CL_int), status)
    return unsafe_ref(convert(Ptr{CL_int}, status))
end

function profiling_info(evt::Event, param::CL_profiling_info)
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
function profiling_info(evt::Event, param::Symbol)
    if     param == :pqueued
    elseif param == :psubmitted
    elseif param == :pstart
    elseif param == :pend 
        return profiling_info(evt, CL_PROFILING_COMMAND_END)
    else
        throw(CLError(CL_INVALID_VALUE))
    end
end


let command_queue(evt::Event) = begin
        cmd_q = Array(CL_command_queue, 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_QUEUE,
                                  sizeof(CL_command_queue), cmd_q, C_NULL)
        return CommandQueue(cmd_q[1])
    end
    
    command_type(evt::Event) = begin
        cmd_t = Array(CL_int , 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_COMMAND_TYPE,
                                  sizeof(CL_int), cmd_t, C_NULL)
        return cmd_t[1]
    end

    reference_count(evt::Event) = begin
        cnt = Array(CL_uint, 1)
        @check api.clGetEventInfo(evt.id, CL_EVENT_REFERENCE_COUNT,
                                  sizeof(CL_uint), cnt, C_NULL)
        return cnt[1]
    end

    context(evt::Event) = begin
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

    function info(q::CommandQueue, qinfo::Symbol)
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
