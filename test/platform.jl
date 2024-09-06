@testset "Platform" begin
    @testset "Info" begin
        @test length(cl.platforms()) == cl.num_platforms()

        @test platform != nothing
        @test pointer(platform) != C_NULL
        for k in [:profile, :version, :name, :vendor, :extensions]
            @test platform[k] == cl.info(platform, k)
        end
        v = opencl_version(platform)
        @test 1 <= v.major <= 3
        @test 0 <= v.minor <= 2
    end

    @testset "Equality" begin
        platform       = cl.platforms()[1]
        platform_copy  = cl.platforms()[1]

        @test pointer(platform) == pointer(platform_copy)
        @test hash(platform) == hash(platform_copy)
        @test isequal(platform, platform)

        if length(cl.platforms()) > 1
            p1 = cl.platforms()[1]
            for p2 in cl.platforms()[2:end]
                @test pointer(p2) != pointer(p1)
                @test hash(p2) != hash(p1)
                @test !isequal(p2, p1)
            end
        end
    end
end
