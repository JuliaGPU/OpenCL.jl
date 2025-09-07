using Random

const n = 256

function apply_seed(seed)
    if seed === missing
        # should result in different numbers across launches
        Random.seed!()
        # XXX: this currently doesn't work, because of the definition in Base,
        #      `seed!(r::MersenneTwister=default_rng())`, which breaks overriding
        #      `default_rng` with a non-MersenneTwister RNG.
    elseif seed !== nothing
        # should result in the same numbers
        Random.seed!(seed)
    elseif seed === nothing
        # should result in different numbers across launches,
        # as determined by the seed set during module loading.
    end
end

eltypes = [filter(x -> !(x <: Complex), GPUArraysTestSuite.supported_eltypes(CLArray)); UInt16; UInt32; UInt64]

@testset "rand($T), seed $seed" for T in eltypes, seed in (nothing, #=missing,=# 1234)
    # different kernel invocations should get different numbers
    @testset "across launches" begin
        function kernel(A::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            tid = get_global_id(1)
            A[tid] = rand(T)
            return nothing
        end

        a = OpenCL.zeros(T, n)
        b = OpenCL.zeros(T, n)

        @opencl global_size=n local_size=n kernel(a, seed)
        @opencl global_size=n local_size=n kernel(b, seed)

        if seed === nothing || seed === missing
            @test Array(a) != Array(b)
        else
            @test Array(a) == Array(b)
        end
    end

    # multiple calls to rand should get different numbers
    @testset "across calls" begin
        function kernel(A::AbstractArray{T}, B::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            tid = get_global_id(1)
            A[tid] = rand(T)
            B[tid] = rand(T)
            return nothing
        end

        a = OpenCL.zeros(T, n)
        b = OpenCL.zeros(T, n)

        @opencl global_size=n local_size=n kernel(a, b, seed)

        @test Array(a) != Array(b)
    end

    # different threads should get different numbers
    @testset "across threads, dim $active_dim" for active_dim in 1:6
        function kernel(A::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            id = get_local_id(1) * get_local_id(2) * get_local_id(3) *
                 get_group_id(1) * get_group_id(2) * get_group_id(3)
            if 1 <= id <= length(A)
                A[id] = rand(T)
            end
            return nothing
        end

        tx, ty, tz, bx, by, bz = [dim == active_dim ? 3 : 1 for dim in 1:6]
        gx, gy, gz = tx*bx, ty*by, tz*bz
        a = OpenCL.zeros(T, 3)

        @opencl local_size=(tx, ty, tz) global_size=(gx, gy, gz) kernel(a, seed)

        # NOTE: we don't just generate two numbers and compare them, instead generating a
        #       couple more and checking they're not all the same, in order to avoid
        #       occasional collisions with lower-precision types (i.e., Float16).
        @test length(unique(Array(a))) > 1
    end
end

@testset "basic randn($T), seed $seed" for T in filter(x -> x <: Base.IEEEFloat, eltypes), seed in (nothing, #=missing,=# 1234)
    function kernel(A::AbstractArray{T}, seed) where {T}
        apply_seed(seed)
        tid = get_global_id(1)
        A[tid] = randn(T)
        return
    end

    a = OpenCL.zeros(T, n)
    b = OpenCL.zeros(T, n)

    @opencl global_size=n local_size=n kernel(a, seed)
    @opencl global_size=n local_size=n kernel(b, seed)

    if seed === nothing || seed === missing
        @test Array(a) != Array(b)
    else
        @test Array(a) == Array(b)
    end
end

@testset "basic randexp($T), seed $seed" for T in filter(x -> x <: Base.IEEEFloat, eltypes), seed in (nothing, #=missing,=# 1234)
    function kernel(A::AbstractArray{T}, seed) where {T}
        apply_seed(seed)
        tid = get_global_id(1)
        A[tid] = randexp(T)
        return
    end

    a = OpenCL.zeros(T, n)
    b = OpenCL.zeros(T, n)

    @opencl global_size=n local_size=n kernel(a, seed)
    @opencl global_size=n local_size=n kernel(b, seed)

    if seed === nothing || seed === missing
        @test Array(a) != Array(b)
    else
        @test Array(a) == Array(b)
    end
end
