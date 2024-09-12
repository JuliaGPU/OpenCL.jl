using LinearAlgebra

@testset "CLArray" begin
    @testset "constructors" begin
        @test CLArray{Float32,1}(undef, 1) isa CLArray{Float32,1}
        @test CLArray{Float32,1}(undef, 1; device=:r) isa CLArray{Float32,1}
        @test CLArray{Float32,1}(undef, 1; host=:r) isa CLArray{Float32,1}

        @test CLArray{Float32}(undef, 1, 2) isa CLArray{Float32,2}
        @test CLArray{Float32}(undef, 1, 2; device=:r) isa CLArray{Float32,2}
        @test CLArray{Float32}(undef, 1, 2; host=:r) isa CLArray{Float32,2}

        @test CLArray{Float32}(undef, (1, 2)) isa CLArray{Float32,2}
        @test CLArray{Float32}(undef, (1, 2); device=:r) isa CLArray{Float32,2}
        @test CLArray{Float32}(undef, (1, 2); host=:r) isa CLArray{Float32,2}

        hostarray = rand(Float32, 128*64)
        A = CLArray(hostarray)
        @test A isa CLArray{Float32,1}
        @test Array(A) == hostarray

        B = CLArray(hostarray; device=:r, host=:rw)
        @test B isa CLArray{Float32,1}
        @test Array(B) == hostarray

        @test Array(copy(A)) == Array(A)
    end

    @testset "fill" begin
        @test Array(OpenCL.fill(Float32(0.5), 32, 64)) == fill(Float32(0.5), 32, 64)
        @test Array(OpenCL.zeros(Float32, 64)) == zeros(Float32, 64)
        @test Array(OpenCL.ones(Float32, 64)) == ones(Float32, 64)
    end

    @testset "core functions" begin
        A = CLArray(rand(Float32, 128, 64))
        @test size(A) == (128, 64)
        @test ndims(A) == 2
        @test length(A) == 128*64

        # reshape
        B = reshape(A, 128*64)
        @test reshape(B, 128, 64) == A

        # transpose
        B = OpenCL.zeros(Float32, 64, 128)
        ev = transpose!(B, A)
        @test Array(A)' == Array(B)
    end
end
