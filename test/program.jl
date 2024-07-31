@testset "Program" begin
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
        ctx = cl.Context(device)
        cl.Program(ctx, source=test_source)
    end

    @testset "source constructor" begin
        ctx = cl.Context(device)
        prg = cl.Program(ctx, source=test_source)
        @test prg != nothing
    end
    @testset "info" begin
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

    @testset "build" begin
        ctx = cl.Context(device)
        prg = cl.Program(ctx, source=test_source)
        @test cl.build!(prg) != nothing

        @test prg[:build_status][device] == cl.CL_BUILD_SUCCESS
        @test prg[:build_log][device] isa String
    end

    @testset "source code" begin
       ctx = cl.Context(device)
       prg = cl.Program(ctx, source=test_source)
       @test prg[:source] == test_source
    end

    if device[:platform][:name] == "Portable Computing Language"
        @warn("Skipping unsupported binary build on POCL")
    else
        @testset "binaries" begin
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
