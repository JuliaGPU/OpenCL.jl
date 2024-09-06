using LinearAlgebra

@testset "CLArray" begin
    @testset "constructors" begin
        ctx = cl.Context(device)
        queue = cl.CmdQueue(ctx)
        hostarray = zeros(Float32, 128*64)
        A = CLArray(queue, hostarray)

        @test CLArray(queue, (:rw, :copy), hostarray) != nothing

        @test CLArray(queue, hostarray, flags=(:rw, :copy)) != nothing

        @test CLArray(queue, hostarray) != nothing

        @test CLArray(cl.Buffer(Float32, ctx, length(hostarray), (:r, :copy), hostbuf=hostarray),
                      queue,
                      (128, 64)) != nothing

        @test copy(A) == A
    end

    @testset "fill" begin
        ctx = cl.Context(device)
        queue = cl.CmdQueue(ctx)

        @test to_host(fill(Float32, queue, Float32(0.5),
                                        32, 64)) == fill(Float32(0.5), 32, 64)
        @test to_host(zeros(Float32, queue, 64)) == zeros(Float32, 64)
        @test to_host(ones(Float32, queue, 64)) == ones(Float32, 64)
    end

    @testset "core functions" begin
        ctx = cl.Context(device)
        queue = cl.CmdQueue(ctx)
        A = CLArray(queue, rand(Float32, 128, 64))
        @test size(A) == (128, 64)
        @test ndims(A) == 2
        @test length(A) == 128*64
        # reshape
        B = reshape(A, 128*64)
        @test reshape(B, 128, 64) == A
        # transpose
        X = CLArray(queue, rand(Float32, 32, 32))
        B = zeros(Float32, queue, 64, 128)
        ev = transpose!(B, A)
        cl.wait(ev)
        #@test to_host(copy(A')) == to_host(B)
    end
end
