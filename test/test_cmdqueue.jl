facts("OpenCL.CmdQueue") do

    context("OpenCL.CmdQueue constructor") do
        has_warned = false
        @fact_throws cl.CmdQueue(nothing, nothing) "error"
        for platform in cl.platforms()
            for device in cl.devices(platform)
                ctx = cl.Context(device)
                @fact cl.CmdQueue(ctx) --> anything "no error"
                @fact cl.CmdQueue(ctx, device) --> anything "no error"
                @fact cl.CmdQueue(ctx, :profile) --> anything "no error"
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
                @fact_throws cl.CmdQueue(ctx, :unrecognized_flag) "error"
                @fact_throws cl.CmdQueue(ctx, device, :unrecognized_flag) "error"
                for flag in [:profile, :out_of_order]
                    @fact_throws cl.CmdQueue(ctx, (flag, :unrecognized_flag)) "error"
                    @fact_throws cl.CmdQueue(ctx, device, (:unrecognized_flag, flag)) "error"
                    @fact_throws cl.CmdQueue(ctx, (flag, flag)) "error"
                    @fact_throws cl.CmdQueue(ctx, device, (flag, flag)) "error"
                end
            end
        end
    end

    context("OpenCL.CmdQueue info") do
        for platform in cl.platforms()
            for device in cl.devices(platform)
                ctx = cl.Context(device)
                q1 = cl.CmdQueue(ctx)
                q2 = cl.CmdQueue(ctx, device)
                for q in (q1, q2)
                    @fact q[:context] --> ctx
                    @fact q[:device] --> device
                    @fact q[:reference_count] > 0 --> true
                    @fact typeof(q[:properties]) --> cl.CL_command_queue_properties
                end
            end
        end
    end
end
