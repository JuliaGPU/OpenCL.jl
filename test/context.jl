
function context_test_callback(arg1, arg2, arg3)
    # We're not really testing it because, nvidia doesn't seem to care about this functionality:
    # https://devtalk.nvidia.com/default/topic/497433/context-callback-never-called/
    OpenCL.cl.log_error("Callback works")
    return
end

function create_context_error(ctx)
    empty_kernel = "
    __kernel void test() {
        int c = 1 + 1;
    };"
    try
        p = cl.Program(ctx, source = empty_kernel) |> cl.build!
        k = cl.Kernel(p, "test")
        q = cl.CmdQueue(ctx)
        q(k, 1, 10000000)
    catch
    end
end


@testset "Context" begin
    @testset "constructor" begin
        @test_throws MethodError (cl.Context([]))
        ctx = cl.Context(device)
        @test ctx != nothing
        ctx_id = ctx.id
        ctx2 = cl.Context(ctx_id)
        @test cl.is_ctx_id_alive(ctx_id)
        @test ctx.id != C_NULL
        @test ctx2.id != C_NULL
        finalize(ctx)
        @test ctx.id == C_NULL
        @test ctx2.id != C_NULL
        @test cl.is_ctx_id_alive(ctx_id)
        finalize(ctx2)
        @test ctx.id == C_NULL
        @test ctx2.id == C_NULL
        # jeez, this segfaults... WHY? I suspect a driver bug for refcount == 0?
        # NVIDIA 381.22
        #@test !cl.is_ctx_id_alive(ctx_id)
        @testset "Context callback" begin
            ctx = cl.Context(device, callback = context_test_callback)
            create_context_error(ctx)
        end
    end


    if platform[:name] == "Portable Computing Language"
        @warn("Skipping OpenCL.Context platform properties for " *
             "Portable Computing Language Platform")
    else
    @testset "platform properties" begin
        try
            cl.Context(cl.CL_DEVICE_TYPE_CPU)
        catch err
            @test typeof(err) == cl.CLError
            # CL_DEVICE_NOT_FOUND could be throw for GPU only drivers
            @test err.desc in (:CL_INVALID_PLATFORM,
                                     :CL_DEVICE_NOT_FOUND)
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

    @testset "create_some_context" begin
        @test cl.create_some_context() != nothing
        @test typeof(cl.create_some_context()) == cl.Context
    end

   @testset "parsing" begin
        properties = [(cl.CL_CONTEXT_PLATFORM, platform)]
        parsed_properties = cl._parse_properties(properties)

        @test isodd(length(parsed_properties))
        @test parsed_properties[end] == 0
        @test parsed_properties[1] == cl.cl_context_properties(cl.CL_CONTEXT_PLATFORM)
        @test parsed_properties[2] == cl.cl_context_properties(platform.id)
    end

end
