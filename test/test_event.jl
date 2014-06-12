using FactCheck
using Base.Test

import OpenCL 
const cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

facts("OpenCL.Event") do

    context("OpenCL.Event status") do
        for platform in cl.platforms()

            if contains(platform[:name], "Portable")
                msg = "Portable Computing Language does not implement User Events"
                warn(msg)
                continue
            end

            for device in cl.devices(platform)
                ctx = cl.Context(device)
                evt = cl.UserEvent(ctx)
                evt[:status]
                @fact evt[:status] => :submitted
                cl.complete(evt)
                @fact evt[:status] => :complete
            end
        end
    end

    context("OpenCL.Event wait") do
        for platform in cl.platforms()

            if contains(platform[:name], "Portable")
                msg = "Portable Computing Language does not implement User Events"
                warn(msg)
                continue
            end

            for device in cl.devices(platform)
                ctx = cl.Context(device)
                # create user event
                usr_evt = cl.UserEvent(ctx)
                q = cl.CmdQueue(ctx)
                cl.enqueue_wait_for_events(q, usr_evt)

                # create marker event
                mkr_evt = cl.enqueue_marker(q)
                
                @fact usr_evt[:status] => :submitted
                @fact cl.cl_event_status(usr_evt[:status]) => cl.CL_SUBMITTED
                @fact mkr_evt[:status] => :queued
                @fact cl.cl_event_status(mkr_evt[:status]) => cl.CL_QUEUED

                cl.complete(usr_evt)
                @fact usr_evt[:status] => :complete
                @fact cl.cl_event_status(usr_evt[:status]) => cl.CL_COMPLETE

                cl.wait(mkr_evt)
                @fact mkr_evt[:status] => :complete
                @fact cl.cl_event_status(mkr_evt[:status]) => cl.CL_COMPLETE
            end
        end
    end

    context("OpenCL.Event callback") do
        for platform in cl.platforms()
            v = cl.opencl_version(platform) 
            if v.major == 1 && v.minor < 1
                info("Skipping OpenCL.Event callback for $(platform[:name]) version < 1.1")
                continue
            end

            if contains(platform[:name], "Portable")
                msg = "Portable Computing Language does not implement User Events"
                warn(msg)
                continue
            end

            for device in cl.devices(platform)
                callback_called = false

                function test_callback(evt, status) 
                    callback_called = true
                    println("Test Callback") 
                end

                ctx = cl.Context(device)
                usr_evt = cl.UserEvent(ctx)
                queue = cl.CmdQueue(ctx)
                
                cl.enqueue_wait_for_events(queue, usr_evt)
                
                mkr_evt = cl.enqueue_marker(queue)
                cl.add_callback(mkr_evt, test_callback)

                @fact usr_evt[:status] => :submitted
                @fact mkr_evt[:status] => :queued
                @fact callback_called => false
                
                cl.complete(usr_evt)
                @fact usr_evt[:status] => :complete
                
                cl.wait(mkr_evt)

                # Give callback some time to finish
                yield()
                sleep(0.5)

                @fact mkr_evt[:status] => :complete
                @fact callback_called => true
            end
        end
    end       
end

