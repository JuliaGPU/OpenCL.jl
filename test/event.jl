if backend in ["POCL", "Intel"]
    # unsupported by POCL
    # hangs on Intel
    @warn "Skipping event tests"
else
@testset "Event" begin
    @testset "status" begin
        ctx = cl.Context(device)
        evt = cl.UserEvent(ctx)
        evt[:status]
        @test evt[:status] == :submitted
        cl.complete(evt)
        @test evt[:status] == :complete
        finalize(evt)
    end

    @testset "wait" begin
        ctx = cl.Context(device)
        # create user event
        usr_evt = cl.UserEvent(ctx)
        q = cl.CmdQueue(ctx)
        cl.enqueue_wait_for_events(q, usr_evt)

        # create marker event
        mkr_evt = cl.enqueue_marker(q)

        @test usr_evt[:status] == :submitted
        @test mkr_evt[:status] in (:queued, :submitted)

        cl.complete(usr_evt)
        @test usr_evt[:status] == :complete

        cl.wait(mkr_evt)
        @test mkr_evt[:status] == :complete

        @test cl.cl_event_status(:running) == cl.CL_RUNNING
        @test cl.cl_event_status(:submitted) == cl.CL_SUBMITTED
        @test cl.cl_event_status(:queued) == cl.CL_QUEUED
        @test cl.cl_event_status(:complete) == cl.CL_COMPLETE
    end

    @testset "callback" begin
        global callback_called = Ref(false)

        function test_callback(evt, status)
            callback_called[] = true
        end

        ctx = cl.Context(device)
        usr_evt = cl.UserEvent(ctx)
        queue = cl.CmdQueue(ctx)

        cl.enqueue_wait_for_events(queue, usr_evt)

        mkr_evt = cl.enqueue_marker(queue)
        cl.add_callback(mkr_evt, test_callback)

        @test usr_evt[:status] == :submitted
        @test mkr_evt[:status] in (:queued, :submitted)
        @test !callback_called[]

        cl.complete(usr_evt)
        @test usr_evt[:status] == :complete

        cl.wait(mkr_evt)

        # Give callback some time to finish
        yield()
        sleep(0.5)

        @test mkr_evt[:status] == :complete
        @test callback_called[]
    end
end
end
