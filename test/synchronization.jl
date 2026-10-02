function slow_fill(a, v)
    i = get_global_id()
    # enough work that the kernel is unlikely to have finished when the launch returns
    acc = 0f0
    for j in 1:10_000
        acc += sin(Float32(j))
    end
    @inbounds a[i] = v + 0f0 * acc
    return
end

# a marker completes once all previously submitted work on the current queue has
marker() = cl.enqueue_marker_with_wait_list(cl.AbstractEvent[])

throwing_kernel(a) = (a[length(a) + 1] = 1f0; return)

@testset "synchronize" begin
    a = OpenCL.zeros(Float32, 1024)
    @opencl global_size=length(a) slow_fill(a, 1f0)
    evt = marker()
    @test OpenCL.synchronize() === nothing
    @test evt.status == :complete
    @test all(==(1f0), Array(a))

    # an explicit queue
    q = cl.CmdQueue()
    evt = cl.queue!(q) do
        @opencl global_size=length(a) slow_fill(a, 2f0)
        marker()
    end
    @test OpenCL.synchronize(q) === nothing
    @test evt.status == :complete
    @test all(==(2f0), Array(a))

    # device-side exceptions are reported
    @opencl throwing_kernel(a)
    @test_throws OpenCL.KernelException OpenCL.synchronize()
end

@testset "@sync" begin
    a = OpenCL.zeros(Float32, 1024)
    evt = OpenCL.@sync begin
        @opencl global_size=length(a) slow_fill(a, 1f0)
        marker()
    end
    @test evt.status == :complete

    # returns the value of the expression, evaluating it only once
    evaluations = Ref(0)
    b = OpenCL.@sync begin
        evaluations[] += 1
        OpenCL.zeros(Float32, 4) .+ 1
    end
    @test evaluations[] == 1
    @test Array(b) == ones(Float32, 4)

    # doesn't clobber the caller's variables
    ret = 1
    x = OpenCL.@sync 2
    @test ret == 1
    @test x == 2

    @test_throws OpenCL.KernelException OpenCL.@sync @opencl throwing_kernel(a)
end

@testset "public API" begin
    if VERSION >= v"1.11.0-DEV.469"
        @test Base.ispublic(OpenCL, :synchronize)
        @test Base.ispublic(OpenCL, Symbol("@sync"))
    end
    @test !Base.isexported(OpenCL, :synchronize)
    @test !Base.isexported(OpenCL, Symbol("@sync"))
end

# a marker that only completes once `gate` is completed. waiting for it deadlocks unless
# waiting yields to the task that completes the gate.
#
# these markers go on a queue of their own: freeing memory waits for the queue it was used
# on, blocking the thread when done by a finalizer, which would deadlock if that queue were
# waiting for a gate that only another task on this thread can complete. for the same
# reason, the only memory used on that queue is kept alive.
const gated_queue = cl.CmdQueue()
const gated_array = cl.queue!(() -> OpenCL.ones(Float32, 4), gated_queue)
function gated_marker()
    gate = cl.UserEvent()
    return gate, cl.enqueue_marker_with_wait_list(cl.AbstractEvent[gate]; queue=gated_queue)
end

# complete `gate` from another task, after a delay that makes waiting go past polling
open_later(gate) = @async (sleep(0.1); cl.complete(gate))

@testset "cooperative waiting" begin
    gate, evt = gated_marker()
    opener = open_later(gate)
    @test wait(evt) === evt
    @test istaskdone(opener)
    @test evt.status == :complete

    gate, evt = gated_marker()
    opener = open_later(gate)
    @test wait(cl.AbstractEvent[evt]) == [evt]
    @test istaskdone(opener)

    gate, evt = gated_marker()
    opener = open_later(gate)
    OpenCL.synchronize(gated_queue)
    @test istaskdone(opener)
    @test evt.status == :complete

    # blocking copies
    gate, evt = gated_marker()
    opener = open_later(gate)
    @test cl.queue!(() -> Array(gated_array), gated_queue) == ones(Float32, 4)
    @test istaskdone(opener)
end

@testset "failed commands" begin
    gate, evt = gated_marker()
    @async (sleep(0.1); cl.clSetUserEventStatus(gate, cl.CL_INVALID_VALUE))
    @test_throws cl.CLError wait(evt)
end

# cancel a task waiting for the device, which is what ^C does on Julia 1.14+. (earlier
# versions throw an `InterruptException` into the task, but that cannot be done reliably
# from Julia code, as it is incorrect to `schedule` a task that has already started.)
if isdefined(Base, :CANCEL_TOKEN)
@testset "cancellation" begin
    # like `Threads.@spawn`, running `f` under a cancellation token that can be cancelled
    function cancellable_task(f)
        src = Base.CancellationTokenSource()
        task = Base.ScopedValues.with(() -> Threads.@spawn(f()),
                                      Base.CANCEL_TOKEN => Base.CancellationToken(src))
        return task, () -> Base.cancel!(src)
    end

    gate, evt = gated_marker()
    task, cancel = cancellable_task(() -> wait(evt))
    sleep(0.1)
    @test !istaskdone(task)
    cancel()
    @test_throws TaskFailedException wait(task)
    @test evt.status != :complete

    # the command keeps executing, and can be waited for again
    opener = open_later(gate)
    @test wait(evt) === evt
    @test istaskdone(opener)

    # blocking transfers only return once they have completed, even when cancelled, as
    # the caller may release the memory they access
    gate, evt = gated_marker()
    dev = cl.device()
    task, cancel = cancellable_task() do
        cl.device!(dev)
        cl.queue!(() -> Array(gated_array), gated_queue)
    end
    sleep(0.1)
    cancel()
    sleep(0.1)
    @test !istaskdone(task)
    cl.complete(gate)
    try
        wait(task)
    catch
    end
    @test istaskdone(task)
end
end
