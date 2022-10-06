@testset "OpenCL.Program" begin

    test_source = "
    __kernel void sum(__global const float *a,
                      __global const float *b,
                      __global float *c)
    {
      uint gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    "

    function create_test_program()
        ctx = cl.create_some_context()
        cl.Program(ctx, source=test_source)
    end

    @testset "OpenCL.Program source constructor" begin
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            @test prg != nothing
        end
    end
    @testset "OpenCL.Program info" begin
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)

            @test prg[:context] == ctx

            @test typeof(prg[:devices]) == Vector{cl.Device}
            @test length(prg[:devices]) > 0
            @test device in prg[:devices]

            @test typeof(prg[:source]) == String
            @test prg[:source] == test_source

            @test prg[:reference_count] > 0
            @test isempty(strip(prg[:build_log][device]))
         end
    end

    @testset "OpenCL.Program build" begin
        for device in cl.devices()
            @testset "Device $(device)" begin
                ctx = cl.Context(device)
                prg = cl.Program(ctx, source=test_source)
                @test cl.build!(prg) != nothing

                # BUILD_SUCCESS undefined in POCL implementation..
                if device[:platform][:name] == "Portable Computing Language"
                    @warn("Skipping OpenCL.Program build for Portable Computing Language Platform")
                    continue
                end
                @test prg[:build_status][device] == cl.CL_BUILD_SUCCESS

                # test build by methods chaining
                @test prg[:build_status][device] == cl.CL_BUILD_SUCCESS
                if device[:platform][:name] != "Intel(R) OpenCL"
                    # The intel CPU driver is very verbose on Linux and output
                    # compilation status even without any warnings
                    @test isempty(strip(prg[:build_log][device]))
                end
            end
        end
    end

    @testset "OpenCL.Program source code" begin
        for device in cl.devices()
           ctx = cl.Context(device)
           prg = cl.Program(ctx, source=test_source)
           @test prg[:source] == test_source
        end
    end

    @testset "OpenCL.Program binaries" begin
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source) |> cl.build!

            @test device in collect(keys(prg[:binaries]))
            binaries = prg[:binaries]
            @test device in collect(keys(binaries))
            @test binaries[device] != nothing
            @test length(binaries[device]) > 0
            prg2 = cl.Program(ctx, binaries=binaries)
            @test prg2[:binaries] == binaries
            try
                prg2[:source]
                error("should not happen")
            catch err
                @test isa(err, cl.CLError)
                @test err.code == -45
                @test err.desc == :CL_INVALID_PROGRAM_EXECUTABLE
            end
        end
    end
end
