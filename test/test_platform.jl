@testset "OpenCL.Platform" begin
    @testset "Platform Info" begin
        @test length(cl.platforms()) == cl.num_platforms()
        for p in cl.platforms()
            @testset "Platform $(p[:profile])" begin
                @test p != nothing
                @test pointer(p) != C_NULL
                for k in [:profile, :version, :name, :vendor, :extensions]
                    @test p[k] == cl.info(p, k)
                end
                v = cl.opencl_version(p)
                @test 1 <= v.major <= 3
                @test 0 <= v.minor <= 2
            end
        end
    end

    @testset "Platform Equality" begin
        platform       = cl.platforms()[1]
        platform_copy  = cl.platforms()[1]

        @test pointer(platform) == pointer(platform_copy)
        @test hash(platform) == hash(platform_copy)
        @test isequal(platform, platform)

        if length(cl.platforms()) > 1
            for p in cl.platforms()[2:end]
                @test pointer(platform) != pointer(p)
                @test hash(platform) != hash(p)
                @test !isequal(platform, p)
            end
        end
    end
end
