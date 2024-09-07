@testset "CmdQueue" begin
    @testset "constructor" begin
        @test_throws MethodError cl.CmdQueue(nothing, nothing)
        @test cl.CmdQueue(cl.context()) != nothing
        @test cl.CmdQueue(cl.context(), cl.device()) != nothing
        @test cl.CmdQueue(cl.context(), :profile) != nothing
        try
            cl.CmdQueue(cl.context(), cl.device(), :out_of_order)
            cl.CmdQueue(cl.context(), cl.device(), (:profile, :out_of_order))
        catch err
            @warn("Platform $(cl.device()[:platform][:name]) does not seem to " *
                  "suport out of order queues: \n$err",maxlog=1,
                  exception=(err, catch_backtrace()))
        end
        @test_throws ArgumentError cl.CmdQueue(cl.context(), cl.device(), :unrecognized_flag)
        for flag in [:profile, :out_of_order]
            @test_throws ArgumentError cl.CmdQueue(cl.context(), (flag, :unrecognized_flag))
            @test_throws ArgumentError cl.CmdQueue(cl.context(), cl.device(), (:unrecognized_flag, flag))
            @test_throws ArgumentError cl.CmdQueue(cl.context(), (flag, flag))
            @test_throws ArgumentError cl.CmdQueue(cl.context(), cl.device(), (flag, flag))
        end
    end

    @testset "info" begin
        q1 = cl.CmdQueue(cl.context())
        q2 = cl.CmdQueue(cl.context(), cl.device())
        for q in (q1, q2)
            @test q[:context] == cl.context()
            @test q[:device] == cl.device()
            @test q[:reference_count] > 0
            @test typeof(q[:properties]) == cl.cl_command_queue_properties
        end
    end
end
