using LinearAlgebra

@testset "CLArray" begin
    @testset "constructors" begin
        hostarray = zeros(Float32, 128*64)
        A = CLArray(hostarray)

        @test CLArray((:rw, :copy), hostarray) != nothing

        @test CLArray(hostarray, flags=(:rw, :copy)) != nothing

        @test CLArray(hostarray) != nothing

        @test CLArray(cl.Buffer(Float32, length(hostarray), (:r, :copy), hostbuf=hostarray),
                      (128, 64)) != nothing

        @test copy(A) == A
    end

    @testset "fill" begin
        @test to_host(OpenCL.fill(Float32, Float32(0.5),
                                        32, 64)) == fill(Float32(0.5), 32, 64)
        @test to_host(OpenCL.zeros(Float32, 64)) == zeros(Float32, 64)
        @test to_host(OpenCL.ones(Float32, 64)) == ones(Float32, 64)
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
        X = CLArray(rand(Float32, 32, 32))
        B = OpenCL.zeros(Float32, 64, 128)
        ev = transpose!(B, A)
        cl.wait(ev)
        #@test to_host(copy(A')) == to_host(B)
    end
end
