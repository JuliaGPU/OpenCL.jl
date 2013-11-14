using FactCheck
using Base.Test

import OpenCL 
const cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

facts("OpenCL.CmdQueue") do 
    
    context("OpenCL.CmdQueue constructor") do
        @fact @throws_pred(cl.CmdQueue(nothing, nothing)) => (true, "error")
        for platform in cl.platforms()
            if platform[:name] == "Portable Computing Language"
                @fact "Portable Computing Language fails on :out_of_order" => true
                continue
            end
            for device in cl.devices(platform)
                ctx = cl.Context(device)
                @fact @throws_pred(cl.CmdQueue(ctx)) => (false, "no error")
                @fact @throws_pred(cl.CmdQueue(ctx, device)) => (false, "no error")
                for flag in [:profile, :out_of_order, (:profile, :out_of_order)]
                    @fact @throws_pred(cl.CmdQueue(ctx, flag)) => (false, "no error")
                    @fact @throws_pred(cl.CmdQueue(ctx, device, flag)) => (false, "no error")
                end
                @fact @throws_pred(cl.CmdQueue(ctx, :unrecognized_flag)) => (true, "error")
                @fact @throws_pred(cl.CmdQueue(ctx, device, :unrecognized_flag)) => (true, "error")
                for flag in [:profile, :out_of_order]
                    @fact @throws_pred(cl.CmdQueue(ctx, (flag, :unrecognized_flag))) => (true, "error")
                    @fact @throws_pred(cl.CmdQueue(ctx, device, (:unrecognized_flag, flag))) => (true, "error")
                    @fact @throws_pred(cl.CmdQueue(ctx, (flag, flag))) => (true, "error")
                    @fact @throws_pred(cl.CmdQueue(ctx, device, (flag, flag))) => (true, "error")
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
                    @fact q[:context] => ctx
                    @fact q[:device] => device
                    @fact q[:reference_count] > 0 => true
                    @fact typeof(q[:properties]) => cl.CL_command_queue_properties
                end
            end
        end
    end
end
