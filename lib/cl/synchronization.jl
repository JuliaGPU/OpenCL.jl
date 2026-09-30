# cooperative synchronization
#
# Waiting for the device should not block the calling thread in the OpenCL driver, but
# yield to the Julia scheduler so that other tasks can run in the meantime. GPUToolbox's
# `cooperative_wait` implements this by polling the status of commands for a while, and
# then handing the blocking wait to a worker thread.

# whether to wait for the device cooperatively. disable to block in the OpenCL driver
# instead, e.g., for bisecting issues or comparing against the blocking behavior.
const nonblocking_synchronization =
    @load_preference("nonblocking_synchronization", true)

event_ids(evts) = cl_event[pointer(evt) for evt in evts]

# commands have completed when their execution status is `CL_COMPLETE` (zero), or negative
# when they were terminated abnormally
function iscomplete(evt::AbstractEvent)
    status = Ref{Cint}()
    clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(Cint), status, C_NULL)
    return status[] <= CL_COMPLETE
end

# commands only need to start executing once the queue they were submitted to has been
# flushed. waiting in the driver flushes implicitly, but polling does not.
function flush_queue(evt::AbstractEvent)
    queue = Ref{cl_command_queue}()
    clGetEventInfo(evt, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), queue, C_NULL)
    # user events do not belong to a queue
    queue[] == C_NULL || clFlush(queue[])
    return
end

# wait for events to complete, throwing a `CLError` if a command was terminated abnormally.
#
# by default, the wait can be interrupted, in which case the commands may still be
# executing. waits that cannot be interrupted only return (or throw) once the commands have
# completed, which is required when they access memory that the caller releases upon
# returning.
function wait_events(evts::Vector{<:AbstractEvent}; cancellable::Bool=true)
    if nonblocking_synchronization
        try
            foreach(flush_queue, evts)
            cooperative_wait(evts; isdone=evts -> all(iscomplete, evts), cancellable) do evts
                GC.@preserve evts unchecked_clWaitForEvents(length(evts), event_ids(evts))
            end
        catch
            # waits that cannot be interrupted only throw once the commands have completed
            # (or waiting failed, in which case this blocks), but host memory still needs to
            # be synchronized before the caller unwinds
            cancellable || GC.@preserve evts begin
                unchecked_clWaitForEvents(length(evts), event_ids(evts))
            end
            rethrow()
        end
    end

    # synchronize host memory and report errors (without blocking, if we waited above)
    GC.@preserve evts clWaitForEvents(length(evts), event_ids(evts))
    return
end

"""
    wait(evt::cl.AbstractEvent)
    wait(evts::Vector{cl.AbstractEvent})

Wait for the commands associated with the event(s) to complete, throwing a `CLError` if a
command was terminated abnormally.

While waiting, the current task yields to others. Waiting is interruptible, in which case
the commands keep executing.
"""
function Base.wait(evt::AbstractEvent)
    wait_events(AbstractEvent[evt])
    return evt
end

function Base.wait(evts::Vector{AbstractEvent})
    isempty(evts) || wait_events(evts)
    return evts
end
