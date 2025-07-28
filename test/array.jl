using LinearAlgebra
import Adapt

@testset "CLArray" begin
    @testset "constructors" begin
        xs = CLArray{Int, 2, cl.Buffer}(undef, 2, 3)
        @test collect(CLArray([1 2; 3 4])) == [1 2; 3 4]
        @test testf(vec, rand(Float32, 5, 3))
        @test Base.elsize(xs) == sizeof(Int)
        @test CLArray{Int, 2}(xs) === xs

        @test device_accessible(xs)
        @test !host_accessible(xs)
        @test_throws ArgumentError Base.unsafe_convert(Ptr{Int}, xs)
        @test_throws ArgumentError Base.unsafe_convert(Ptr{Float32}, xs)

        @test collect(OpenCL.zeros(Float32, 2, 2)) == zeros(Float32, 2, 2)
        @test collect(OpenCL.ones(Float32, 2, 2)) == ones(Float32, 2, 2)

        @test collect(OpenCL.fill(0, 2, 2)) == zeros(Int, 2, 2)
        @test collect(OpenCL.fill(1, 2, 2)) == ones(Int, 2, 2)
    end

    @testset "adapt" begin
        A = rand(Float32, 3, 3)
        dA = CLArray(A)
        @test Adapt.adapt(Array, dA) == A
        @test Adapt.adapt(CLArray, A) isa CLArray
        @test Array(Adapt.adapt(CLArray, A)) == A
    end

    @testset "reshape" begin
        A = [
            1 2 3 4
            5 6 7 8
        ]
        gA = reshape(CLArray(A), 1, 8)
        _A = reshape(A, 1, 8)
        _gA = Array(gA)
        @test all(_A .== _gA)
        A = [1, 2, 3, 4]
        gA = reshape(CLArray(A), 4)
    end

    @testset "fill(::SubArray)" begin
        xs = OpenCL.zeros(Float32, 3)
        fill!(view(xs, 2:2), 1)
        @test Array(xs) == [0, 1, 0]
    end
    # TODO: Look into how to port the @sync

    if cl.usm_supported(cl.device())
        @testset "shared buffers & unsafe_wrap" begin
            a = CLVector{Int, cl.UnifiedSharedMemory}(undef, 2)

            # check that basic operations work on arrays backed by shared memory
            fill!(a, 40)
            a .+= 2
            @test Array(a) == [42, 42]

            # derive an Array object and test that the memory keeps in sync
            b = unsafe_wrap(Array, a)
            b[1] = 100
            @test Array(a) == [100, 42]
            copyto!(a, 2, [200], 1, 1)
            cl.finish(cl.queue())
            @test b == [100, 200]
        end

        # https://github.com/JuliaGPU/CUDA.jl/issues/2191
        @testset "preserving memory types" begin
            a = CLVector{Int, cl.UnifiedSharedMemory}([1])
            @test OpenCL.memtype(a) == cl.UnifiedSharedMemory

            # unified-ness should be preserved
            b = a .+ 1
            @test OpenCL.memtype(b) == cl.UnifiedSharedMemory

            # when there's a conflict, we should defer to unified memory
            c = CLVector{Int, cl.UnifiedSharedMemory}([1])
            d = CLVector{Int, cl.UnifiedDeviceMemory}([1])
            e = c .+ d
            @test OpenCL.memtype(e) == cl.UnifiedSharedMemory
        end
    else
        @warn "Skipping USM-specific tests as not supported on device $(cl.device())"
    end

    @testset "resizing" begin
        a = CLArray([1, 2, 3])

        resize!(a, 3)
        @test length(a) == 3
        @test Array(a) == [1, 2, 3]

        resize!(a, 5)
        @test length(a) == 5
        @test Array(a)[1:3] == [1, 2, 3]

        resize!(a, 2)
        @test length(a) == 2
        @test Array(a)[1:2] == [1, 2]

        b = CLArray{Int}(undef, 0)
        @test length(b) == 0
        resize!(b, 1)
        @test length(b) == 1
    end

    @testset "broadcasting" begin
        a = rand(Float32, 2, 3)
        b = rand(Float32, 2)

        c = a .+ b
        a_cl, b_cl = CLArray(a), CLArray(b)
        c_cl = a_cl .+ b_cl
        @test Array(c_cl) == c
        @test c_cl isa CLArray{Float32, 2, cl.UnifiedDeviceMemory}

        if cl.usm_supported(cl.device())
            a_cl, b_cl = CLMatrix{Float32, cl.UnifiedSharedMemory}(a), CLVector{Float32, OpenCL.cl.UnifiedDeviceMemory}(b)
            c_cl = a_cl .+ b_cl
            @test Array(c_cl) == c
            @test c_cl isa CLArray{Float32, 2, cl.UnifiedSharedMemory}
        end
    end
end
