@testset "OpenCL.Minver" begin
    @testset "OpenCL.Minver platform" begin
        version = cl.opencl_version(platform)

        v11 = cl.min_v11(platform)
        v12 = cl.min_v12(platform)
        v20 = cl.min_v20(platform)
        v21 = cl.min_v21(platform)
        v22 = cl.min_v22(platform)

        @test v11 == (version >= v"1.1")
        @test v12 == (version >= v"1.2")
        @test v20 == (version >= v"2.0")
        @test v21 == (version >= v"2.1")
        @test v22 == (version >= v"2.2")
    end

    @testset "OpenCL.Minver device" begin
        version = cl.opencl_version(device)

        v11 = cl.min_v11(device)
        v12 = cl.min_v12(device)
        v20 = cl.min_v20(device)
        v21 = cl.min_v21(device)
        v22 = cl.min_v22(device)

        @test v11 == (version >= v"1.1")
        @test v12 == (version >= v"1.2")
        @test v20 == (version >= v"2.0")
        @test v21 == (version >= v"2.1")
        @test v22 == (version >= v"2.2")
    end
end
