@testset "CmdQueue" begin
    @testset "constructor" begin
        @test cl.CmdQueue() != nothing
        @test cl.CmdQueue(:profile) != nothing
        try
            cl.CmdQueue(:out_of_order)
            cl.CmdQueue((:profile, :out_of_order))
        catch err
            @warn("Platform $(cl.device()[:platform][:name]) does not seem to " *
                  "suport out of order queues: \n$err",maxlog=1,
                  exception=(err, catch_backtrace()))
        end
        @test_throws ArgumentError cl.CmdQueue(:unrecognized_flag)
        for flag in [:profile, :out_of_order]
            @test_throws ArgumentError cl.CmdQueue((flag, :unrecognized_flag))
            @test_throws ArgumentError cl.CmdQueue((flag, flag))
        end
    end

    @testset "info" begin
        q = cl.CmdQueue()
        @test q[:context] == cl.context()
        @test q[:device] == cl.device()
        @test q[:reference_count] > 0
        @test typeof(q[:properties]) == cl.cl_command_queue_properties
    end
end
