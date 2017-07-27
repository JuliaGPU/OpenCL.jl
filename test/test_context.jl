@testset "OpenCL.Context" begin
    @testset "OpenCL.Context constructor" begin
        @test_throws MethodError (cl.Context([]))
        for platform in cl.platforms()
            for device in cl.devices(platform)
                ctx = cl.Context(device)
                @test ctx != nothing
                ctx2 = cl.Context(device)
                # seems like OpenCL doesn't require contexts from the same device
                # to return the same context pointer... But might also happen.
                if ctx.id === ctx2.id
                    # test that we have exactly the same context
                    # important for finalizers to work without double frees
                    @test ctx2 === ctx
                    finalize(ctx)
                    @test ctx.id == C_NULL
                    @test ctx2.id == C_NULL
                end
            end
        end
    end

    @testset "OpenCL.Context platform properties" begin
        for platform in cl.platforms()
            try
                cl.Context(cl.CL_DEVICE_TYPE_CPU)
            catch err
                @test typeof(err) == cl.CLError
                # CL_DEVICE_NOT_FOUND could be throw for GPU only drivers
                @test err.desc in (:CL_INVALID_PLATFORM,
                                         :CL_DEVICE_NOT_FOUND)
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
                @test cl.Context(sym_dev_type, properties=properties) != nothing
                @test cl.Context(cl_dev_type, properties=properties) != nothing
                ctx = cl.Context(cl_dev_type, properties=properties)
                @test isempty(cl.properties(ctx)) == false
                test_properties = cl.properties(ctx)

                @test test_properties == properties

                platform_in_properties = false
                for (t, v) in test_properties
                    if t == cl.CL_CONTEXT_PLATFORM
                        @test v[:name] == platform[:name]
                        @test v == platform
                        platform_in_properties = true
                        break
                    end
                end
                @test platform_in_properties
            end
            try
                ctx2 = cl.Context(cl.CL_DEVICE_TYPE_ACCELERATOR,
                                  properties=properties)
            catch err
                @test typeof(err) == cl.CLError
                @test err.desc == :CL_DEVICE_NOT_FOUND
            end
        end
    end

    @testset "OpenCL.Context create_some_context" begin
        @test cl.create_some_context() != nothing
        @test typeof(cl.create_some_context()) == cl.Context
    end

   @testset "OpenCL.Context parsing" begin
        for platform in cl.platforms()
            properties = [(cl.CL_CONTEXT_PLATFORM, platform)]
            parsed_properties = cl._parse_properties(properties)

            @test isodd(length(parsed_properties))
            @test parsed_properties[end] == 0
            @test parsed_properties[1] == cl.cl_context_properties(cl.CL_CONTEXT_PLATFORM)
            @test parsed_properties[2] == cl.cl_context_properties(platform.id)
        end
    end

end
