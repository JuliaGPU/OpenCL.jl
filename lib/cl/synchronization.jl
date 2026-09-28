# cooperative synchronization
#
# Like CUDA.jl, busy-wait briefly for short operations, then block in the driver on a
# separate thread, so that the waiting task yields to the Julia scheduler, and the thread
# it runs on can do other work and take part in garbage collection.

using GPUToolbox: @gcsafe_ccall

# whether the command associated with `evt` has completed. errors count as completion:
# waiting for the event reports them.
function isdone(evt::AbstractEvent)
    status = Ref{Cint}()
    clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(Cint), status, C_NULL)
    return status[] <= CL_COMPLETE
end

# before waiting on another thread, which has some overhead, busy-wait for the event,
# initially without even yielding to other tasks. returns whether the event completed.
function spinning_wait(evt::AbstractEvent)
    isdone(evt) && return true
    for spins in 1:256
        if spins <= 32
            ccall(:jl_cpu_pause, Cvoid, ())
            # allow the GC to run while we're spinning
            ccall(:jl_gc_safepoint, Cvoid, ())
        else
            yield()
        end
        isdone(evt) && return true
    end
    return false
end

struct SyncRequest
    event::AbstractEvent
    status::Base.RefValue{cl_int}
    done::Base.Event
end

const MAX_SYNC_THREADS = 4
const sync_channels = Vector{Channel{SyncRequest}}(undef, MAX_SYNC_THREADS)
const sync_channel_cursor = Threads.Atomic{UInt32}(1)
const sync_channel_lock = ReentrantLock()

# runs on a thread of its own, waiting for the events it's sent
function synchronization_worker(data::Ptr{Cvoid})
    chan = sync_channels[Int(data)]
    while true
        req = take!(chan)
        GC.@preserve req begin
            id = Ref(req.event.id)
            req.status[] = @gcsafe_ccall libopencl.clWaitForEvents(1::cl_uint,
                                                                   id::Ptr{cl_event})::cl_int
        end
        notify(req.done)
    end
end

@noinline function sync_channel(i::Int)
    @lock sync_channel_lock begin
        isassigned(sync_channels, i) && return sync_channels[i]
        chan = Channel{SyncRequest}(Inf)
        sync_channels[i] = chan

        # we don't know the size of uv_thread_t, so reserve enough space
        tid = Ref{NTuple{32, UInt8}}(ntuple(_ -> 0x00, 32))
        cb = @cfunction(synchronization_worker, Cvoid, (Ptr{Cvoid},))
        err = @ccall uv_thread_create(tid::Ptr{Cvoid}, cb::Ptr{Cvoid}, Ptr{Cvoid}(i)::Ptr{Cvoid})::Cint
        err == 0 || Base.uv_error("uv_thread_create", err)
        err = @ccall uv_thread_detach(tid::Ptr{Cvoid})::Cint
        err == 0 || Base.uv_error("uv_thread_detach", err)
        return chan
    end
end

# wait for `evt` on a worker thread, while the calling task yields
function nonblocking_wait(evt::AbstractEvent)
    # sticky per task, so that a task keeps using the same worker, while concurrent tasks
    # spread over the workers
    i = get!(task_local_storage(), :CLSyncChannel) do
        mod1(Int(Threads.atomic_add!(sync_channel_cursor, UInt32(1))), MAX_SYNC_THREADS)
    end::Int
    chan = isassigned(sync_channels, i) ? sync_channels[i] : sync_channel(i)

    req = SyncRequest(evt, Ref{cl_int}(CL_SUCCESS), Base.Event())
    put!(chan, req)
    wait(req.done)
    req.status[] == CL_SUCCESS || throw(CLError(req.status[]))
    return
end

"""
    cl.wait_cooperatively(q::CmdQueue)

Wait for all commands queued on `q` to complete, while letting other tasks run. This does
not check for errors or device-side exceptions, `cl.finish` does.
"""
function wait_cooperatively(q::CmdQueue)
    evt = Ref{cl_event}()
    clEnqueueMarkerWithWaitList(q, 0, C_NULL, evt)
    marker = Event(evt[])
    # the marker only completes once the queue has been submitted to the device
    clFlush(q)
    spinning_wait(marker) || nonblocking_wait(marker)
    return
end
