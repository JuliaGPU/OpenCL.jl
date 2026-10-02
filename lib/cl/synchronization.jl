# cooperative synchronization
#
# Waiting for the device should not block the calling thread in the OpenCL driver, but
# yield to the Julia scheduler so that other tasks can run in the meantime. GPUToolbox's
# `cooperative_wait` implements this, in one of two ways:
# - for GPUs, by polling the status of commands for a while, and then handing a blocking
#   wait to a worker thread;
# - for devices that execute on the host's CPU cores, by polling only briefly, and then
#   having the driver notify us when commands complete. waking a worker, or polling for
#   longer, would compete with the commands for those cores, slowing them down
#   considerably. (drivers for GPUs may deliver these notifications late, NVIDIA's by
#   about 20 ms.)

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
# flushed. waiting in the driver flushes implicitly, but polling does not. returns the
# queue, or `C_NULL` for user events, which do not belong to one.
function flush_queue(evt::AbstractEvent)
    queue = Ref{cl_command_queue}()
    clGetEventInfo(evt, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), queue, C_NULL)
    queue[] == C_NULL || clFlush(queue[])
    return queue[]
end

# whether commands on `queue` execute on the host's CPU cores
const cpu_devices = Dict{cl_device_id, Bool}()
const cpu_devices_lock = ReentrantLock()
function executes_on_cpu(queue::cl_command_queue)
    queue == C_NULL && return false
    device = Ref{cl_device_id}()
    clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), device, C_NULL)
    return @lock cpu_devices_lock get!(cpu_devices, device[]) do
        type = Ref{cl_device_type}()
        clGetDeviceInfo(device[], CL_DEVICE_TYPE, sizeof(cl_device_type), type, C_NULL)
        type[] & CL_DEVICE_TYPE_CPU != 0
    end
end

# driver notification that a command has completed
function notify_completion(::cl_event, ::Cint, payload::Ptr{Cvoid})
    GPUToolbox.signal_completion(payload)
    return
end
subscribe_completion(evt, payload) =
    clSetEventCallback(evt, CL_COMPLETE,
                       @cfunction(notify_completion, Cvoid, (cl_event, Cint, Ptr{Cvoid})),
                       payload)

# block until the events have completed, without checking for errors
blocking_wait(evts::Vector{<:AbstractEvent}) =
    GC.@preserve evts unchecked_clWaitForEvents(length(evts), event_ids(evts))
blocking_wait(evt::AbstractEvent) =
    GC.@preserve evt unchecked_clWaitForEvents(1, Ref(pointer(evt)))

# wait for events to complete, throwing a `CLError` if a command was terminated abnormally.
#
# by default, the wait can be interrupted, in which case the commands may still be
# executing. waits that cannot be interrupted only return (or throw) once the commands have
# completed, which is required when they access memory that the caller releases upon
# returning.
function wait_events(evts::Vector{<:AbstractEvent}; cancellable::Bool=true)
    if nonblocking_synchronization
        try
            on_cpu = true
            for evt in evts
                on_cpu &= executes_on_cpu(flush_queue(evt))
            end
            if on_cpu
                for evt in evts
                    cooperative_wait(blocking_wait, evt; subscribe=subscribe_completion,
                                     isdone=iscomplete, spin=10e-6, cancellable)
                end
            else
                cooperative_wait(blocking_wait, evts; isdone=evts -> all(iscomplete, evts),
                                 cancellable)
            end
        catch
            # waits that cannot be interrupted only throw once the commands have completed
            # (or waiting failed, in which case this blocks), but host memory still needs to
            # be synchronized before the caller unwinds
            cancellable || blocking_wait(evts)
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

# wait for the commands submitted to `queue` so far to complete. this does not check for
# device-side exceptions, as `finish` does.
function wait_idle(queue::CmdQueue; cancellable::Bool=true)
    if nonblocking_synchronization
        # there is no way to query whether a queue is idle, so wait for a marker instead
        marker = enqueue_marker_with_wait_list(AbstractEvent[]; queue)
        wait_events([marker]; cancellable)
    else
        clFinish(queue)
    end
    return
end
