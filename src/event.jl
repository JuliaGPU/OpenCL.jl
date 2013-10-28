# --- low level OpenCL Event ---

const status_dict = (CL_int => Symbol)[CL_QUEUED => :queued,
                                       CL_SUBMITTED => :submitted,
                                       CL_RUNNING => :running,
                                       CL_COMPLETE => :complete]
immutable Event
    id :: CL_event
end

@ocl_func(clWaitForEvents, (CL_uint, Ptr{CL_event}))

#TODO: see if there is a better way to do this
function wait(evt::Event)
    evt_id = Array(CL_event, 1)
    evt_id[1] = evt.id
    clWaitForEvents(1, evt_id)
end
    
@ocl_func(clGetEventInfo, (CL_event, CL_event_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

function status(evt::Event)
    status = CL_int[0]
    clGetEventInfo(evt.id, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(CL_int), status)
    return status_dict[status[1]]
end

@ocl_func(clReleaseEvent, (CL_event,))

function free!(evt::Event)
    if evt.id != C_NULL
        clReleaseEvent(evt.id)
        evt.id = C_NULL
    end 
end 
