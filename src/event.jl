# --- low level OpenCL Event ---

const status_dict = (CL_int => Symbol)[CL_QUEUED => :queued,
                                       CL_SUBMITTED => :submitted,
                                       CL_RUNNING => :running,
                                       CL_COMPLETE => :complete]
immutable Event
    id :: CL_event
end

pointer(evt::Event) = evt.id

#TODO: user events 

@ocl_func(clWaitForEvents, (CL_uint, Ptr{CL_event}))

#TODO: see if there is a better way to do this
function wait(evt::Event)
    evt_id = Array(CL_event, 1)
    evt_id[1] = evt.id
    clWaitForEvents(1, evt_id)
end

function wait(evts::Vector{Event})
    for evt in evts
        wait(evt)
    end
end

function wait(evts::Tuple{N,Event})
    for i in 1:N
        wait(evts[i])
    end
end

@ocl_func(clGetEventInfo, (CL_event, CL_event_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

function status(evt::Event)
    status = CL_int[0]
    clGetEventInfo(evt.id, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(CL_int), status)
    return status_dict[status[1]]
end

function status(evt_id::CL_event)
    status = convert(Ptr{Void}, Array(CL_int, 1))
    clGetEventInfo(evt_id, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(CL_int), status)
    return unsafe_ref(convert(Ptr{CL_int}, status))
end

function clGetEventProfilingInfo(evt_id::CL_event,
                                 param::CL_profiling_info,
                                 param_size::Csize_t,
                                 param_val::Ptr{Void},
                                 param_value_size_ret::Ptr{Csize_t})
    err = ccall((:clGetEventProfilingInfo, libopencl),
                CL_int,
                (CL_event, CL_profiling_info, Csize_t, Ptr{Void}, Ptr{Csize_t}),
                evt_id, param, param_size, param_val, param_value_size_ret)
    if err != CL_SUCCESS
        if err == CL_PROFILING_INFO_NOT_AVAILABLE
            if status(evt_id) != CL_COMPLETE
                #TODO: throw
                error("Event must have :completed status before it can be profiled")
            else
                error("Queue must be created with profile=True")
            end
        error("OpenCL clGetEventProfilingInfo error...")
    end
    return status
end

function profile_start(evt::Event)
    status = convert(Ptr{Void}, Array(CL_ulong, 1))
    clGetEventProfilingInfo(evt.id, CL_PROFILING_COMMAND_START, sizeof(CL_ulong), status, C_NULL)
    return unsafe_ref(convert(Ptr{CL_ulong}, status))
end

function profile_end(evt::Event)
    status = convert(Ptr{Void}, Array(CL_ulong, 1))
    clGetEventProfilingInfo(evt.id, CL_PROFILING_COMMAND_END, sizeof(CL_ulong), status, C_NULL)
    return unsafe_ref(convert(Ptr{CL_ulong}, status))
end

function duration(evt::Event)
    profile_end(evt) - profile_start(evt)
end

function profile_queued(evt::Event)
    status = convert(Ptr{Void}, Array(CL_ulong, 1))
    clGetEventProfilingInfo(evt.id, CL_PROFILING_COMMAND_QUEUED, sizeof(CL_ulong), status, C_NULL)
    return unsafe_ref(convert(Ptr{CL_ulong}, status))
end

function profile_submitted(evt::Event)
    status = convert(Ptr{Void}, Array(CL_ulong, 1))
    clGetEventProfilingInfo(evt.id, CL_PROFILING_COMMAND_SUBMIT, sizeof(CL_ulong), status, C_NULL)
    return unsafe_ref(convert(Ptr{CL_ulong}, status))
end

# TODO: Register Callbacks
# TODO: Julia Events (tie into uv event framework???)

@ocl_func(clReleaseEvent, (CL_event,))

function free!(evt::Event)
    if evt.id != C_NULL
        clReleaseEvent(evt.id)
        evt.id = C_NULL
    end 
end 
