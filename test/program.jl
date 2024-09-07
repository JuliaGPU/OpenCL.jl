@testset "Program" begin
    let
        @test_throws ArgumentError cl.Program(cl.context())
        @test_throws ArgumentError cl.Program(cl.context(); source="", il="")
    end

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
        cl.Program(cl.context(), source=test_source)
    end

    @testset "source constructor" begin
        prg = cl.Program(cl.context(), source=test_source)
        @test prg != nothing
    end
    @testset "info" begin
        prg = cl.Program(cl.context(), source=test_source)

        @test prg[:context] == cl.context()

        @test typeof(prg[:devices]) == Vector{cl.Device}
        @test length(prg[:devices]) > 0
        @test cl.device() in prg[:devices]

        @test typeof(prg[:source]) == String
        @test prg[:source] == test_source

        @test prg[:reference_count] > 0
        @test isempty(strip(prg[:build_log][cl.device()]))
    end

    @testset "build" begin
        prg = cl.Program(cl.context(), source=test_source)
        @test cl.build!(prg) != nothing

        @test prg[:build_status][cl.device()] == cl.CL_BUILD_SUCCESS
        @test prg[:build_log][cl.device()] isa String
    end

    @testset "source code" begin
       prg = cl.Program(cl.context(), source=test_source)
       @test prg[:source] == test_source
    end

    if backend == "POCL"
        @warn "Skipping binary program tests"
    else
        @testset "binaries" begin
            prg = cl.Program(cl.context(), source=test_source) |> cl.build!

            @test cl.device() in collect(keys(prg[:binaries]))
            binaries = prg[:binaries]
            @test cl.device() in collect(keys(binaries))
            @test binaries[cl.device()] != nothing
            @test length(binaries[cl.device()]) > 0
            prg2 = cl.Program(cl.context(), binaries=binaries)
            @test prg2[:binaries] == binaries
            @test prg2[:source] === nothing
        end
    end
end
