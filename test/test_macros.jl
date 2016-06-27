@testset "OpenCL.Macros" begin
    @testset "OpenCL.Macros version platform" begin
        for platform in cl.platforms()

            version = cl.opencl_version(platform)

            v11 = cl.@min_v11? platform true : false
            v12 = cl.@min_v12? platform true : false
            v20 = cl.@min_v20? platform true : false

            if version == v"1.0"
                @test v11 == false
                @test v12 == false
                @test v20 == false
            elseif version == v"1.1"
                @test v11 == true
                @test v12 == false
                @test v20 == false
            elseif version == v"1.2"
                @test v11 == true
                @test v12 == true
                @test v20 == false
            elseif version == v"2.0"
                @test v11 == true
                @test v12 == true
                @test v20 == true
            end
        end
    end

    @testset "OpenCL.Macros version device" begin
        for platform in cl.platforms()
            for device in cl.devices(platform)
                version = cl.opencl_version(device)

                v11 = cl.@min_v11? device true : false
                v12 = cl.@min_v12? device true : false
                v20 = cl.@min_v20? device true : false

                if version == v"1.0"
                    @test v11 == false
                    @test v12 == false
                    @test v20 == false
                elseif version == v"1.1"
                    @test v11 == true
                    @test v12 == false
                    @test v20 == false
                elseif version == v"1.2"
                    @test v11 == true
                    @test v12 == true
                    @test v20 == false
                elseif version == v"2.0"
                    @test v11 == true
                    @test v12 == true
                    @test v20 == true
                end
            end
        end
    end
end

