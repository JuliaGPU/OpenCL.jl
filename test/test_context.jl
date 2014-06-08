using FactCheck
using Base.Test

import OpenCL 
const cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

facts("OpenCL.Context") do

    context("OpenCL.Context constructor") do
        @fact @throws_pred(cl.Context([])) => (true, "error")
        for platform in cl.platforms()
            for device in cl.devices(platform)
                @fact @throws_pred(cl.Context(device)) => (false, "no error")
            end
        end
    end

    context("OpenCL.Context platform properties") do
        for platform in cl.platforms()
            try
                cl.Context(cl.CL_DEVICE_TYPE_CPU)
            catch err
                @fact typeof(err) => cl.CLError
                @fact err.desc => :CL_INVALID_PLATFORM
            end
            
            if platform[:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Context platform properties for " * 
                     "Portable Computing Language Platform")
                continue
            end

            properties = [(cl.CL_CONTEXT_PLATFORM, platform)]
            for (cl_dev_type, sym_dev_type) in [(cl.CL_DEVICE_TYPE_CPU, :cpu),
                                                (cl.CL_DEVICE_TYPE_GPU, :gpu)]
                if !cl.has_device_type(platform, sym_dev_type)
                    continue
                end
                @fact @throws_pred(cl.Context(sym_dev_type, properties=properties)) => (false, "no error")
                @fact @throws_pred(cl.Context(cl_dev_type, properties=properties)) => (false, "no error") 
                ctx = cl.Context(cl_dev_type, properties=properties)
                @fact isempty(cl.properties(ctx)) => false
                test_properties = cl.properties(ctx)

                @fact test_properties => properties

                platform_in_properties = false 
                for (t, v) in test_properties
                    if t == cl.CL_CONTEXT_PLATFORM
                        @fact v[:name] => platform[:name]
                        @fact v == platform => true
                        platform_in_properties = true
                        break
                    end
                end
                @fact platform_in_properties => true 
            end
            try
                ctx2 = cl.Context(cl.CL_DEVICE_TYPE_ACCELERATOR,
                                  properties=properties)
            catch err
                @fact typeof(err) => cl.CLError
                @fact err.desc => :CL_DEVICE_NOT_FOUND
            end
        end
    end

    context("OpenCL.Context create_some_context") do
        @fact @throws_pred(cl.create_some_context()) => (false, "no error")
        @fact typeof(cl.create_some_context()) => cl.Context
    end

    context("OpenCL.Context parsing") do
        for platform in cl.platforms()
            properties = [(cl.CL_CONTEXT_PLATFORM, platform)]
            parsed_properties = cl._parse_properties(properties)

            @fact length(parsed_properties) => isodd
            @fact parsed_properties[end] => 0
            @fact parsed_properties[1] => cl.cl_context_properties(cl.CL_CONTEXT_PLATFORM)
            @fact parsed_properties[2] => cl.cl_context_properties(platform.id)
        end
    end
end

