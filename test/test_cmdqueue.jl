@testset "OpenCL.CmdQueue" begin
    @testset "OpenCL.CmdQueue constructor" begin
        has_warned = false
        @test_throws MethodError cl.CmdQueue(nothing, nothing)
        for platform in cl.platforms()
            for device in cl.devices(platform)
                ctx = cl.Context(device)
                @test cl.CmdQueue(ctx) != nothing
                @test cl.CmdQueue(ctx, device) != nothing
                @test cl.CmdQueue(ctx, :profile) != nothing
                try
                    cl.CmdQueue(ctx, device, :out_of_order)
                    cl.CmdQueue(ctx, device, (:profile, :out_of_order))
                catch err
                    if !has_warned
                        warn("Platform $(device[:platform][:name]) does not seem to " *
                             "suport out of order queues: \n$err")
                        has_warned = true
                    end
                end
                @test_throws ArgumentError cl.CmdQueue(ctx, device, :unrecognized_flag)
                for flag in [:profile, :out_of_order]
                    @test_throws ArgumentError cl.CmdQueue(ctx, (flag, :unrecognized_flag))
                    @test_throws ArgumentError cl.CmdQueue(ctx, device, (:unrecognized_flag, flag))
                    @test_throws ArgumentError cl.CmdQueue(ctx, (flag, flag))
                    @test_throws ArgumentError cl.CmdQueue(ctx, device, (flag, flag))
                end
            end
        end
    end

    @testset "OpenCL.CmdQueue info" begin
        for platform in cl.platforms()
            for device in cl.devices(platform)
                ctx = cl.Context(device)
                q1 = cl.CmdQueue(ctx)
                q2 = cl.CmdQueue(ctx, device)
                for q in (q1, q2)
                    @test q[:context] == ctx
                    @test q[:device] == device
                    @test q[:reference_count] > 0
                    @test typeof(q[:properties]) == cl.CL_command_queue_properties
                end
            end
        end
    end
end
