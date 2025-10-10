using SIMD

function call_on_device(f, args...)
    function kernel(res, f, args...)
        res[] = f(args...)
        return
    end
    T = OpenCL.code_typed(() -> f(args...), ())[][2]
    res = CLArray{T, 0}(undef)
    @opencl kernel(res, f, args...)
    return OpenCL.@allowscalar res[]
end

const float_types = filter(x -> x <: Base.IEEEFloat, GPUArraysTestSuite.supported_eltypes(CLArray))
const ispocl = cl.platform().name == "Portable Computing Language"
const simd_ns = [2, 3, 4, 8, 16]

@testset "intrinsics" begin

@testset "barrier" begin

@on_device barrier(OpenCL.LOCAL_MEM_FENCE)
@on_device barrier(OpenCL.GLOBAL_MEM_FENCE)
@on_device barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)

@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE)
@on_device work_group_barrier(OpenCL.GLOBAL_MEM_FENCE)
@on_device work_group_barrier(OpenCL.IMAGE_MEM_FENCE)

@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.IMAGE_MEM_FENCE)
@on_device work_group_barrier(OpenCL.GLOBAL_MEM_FENCE | OpenCL.LOCAL_MEM_FENCE | OpenCL.IMAGE_MEM_FENCE)

@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_work_item)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_work_group)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_device)
cl.memory_backend() isa cl.SVMBackend && @on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_all_svm_devices)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_sub_group)

end

@testset "mem_fence" begin

@on_device mem_fence(OpenCL.LOCAL_MEM_FENCE)
@on_device mem_fence(OpenCL.GLOBAL_MEM_FENCE)
@on_device mem_fence(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)

@on_device read_mem_fence(OpenCL.LOCAL_MEM_FENCE)
@on_device read_mem_fence(OpenCL.GLOBAL_MEM_FENCE)
@on_device read_mem_fence(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)

@on_device write_mem_fence(OpenCL.LOCAL_MEM_FENCE)
@on_device write_mem_fence(OpenCL.GLOBAL_MEM_FENCE)
@on_device write_mem_fence(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)

end

@testset "atomic_work_item_fence" begin

@on_device atomic_work_item_fence(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_order_relaxed, OpenCL.memory_scope_work_item)
@on_device atomic_work_item_fence(OpenCL.GLOBAL_MEM_FENCE, OpenCL.memory_order_acquire, OpenCL.memory_scope_work_group)
@on_device atomic_work_item_fence(OpenCL.IMAGE_MEM_FENCE, OpenCL.memory_order_release, OpenCL.memory_scope_device)
cl.memory_backend() isa cl.SVMBackend && @on_device atomic_work_item_fence(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_order_acq_rel, OpenCL.memory_scope_all_svm_devices)
@on_device atomic_work_item_fence(OpenCL.GLOBAL_MEM_FENCE, OpenCL.memory_order_seq_cst, OpenCL.memory_scope_sub_group)
@on_device atomic_work_item_fence(OpenCL.IMAGE_MEM_FENCE | OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_order_acquire, OpenCL.memory_scope_sub_group)

end

@testset "math" begin

@testset "unary - $T" for T in float_types
    @testset "$f" for f in [
            acos, acosh,
            asin, asinh,
            atan, atanh,
            cbrt,
            ceil,
            cos, cosh, cospi,
            exp, exp2, exp10, expm1,
            abs,
            floor,
            log, log2, log10, log1p,
            round,
            sin, sinh, sinpi,
            sqrt,
            tan, tanh, tanpi,
            trunc,
        ]
        x = rand(T)
        if f == acosh
            x += 1
        end
        broken = ispocl && T == Float16 && f in [acosh, asinh, atanh, cbrt, cospi, expm1, log1p, sinpi, tanpi]
        @test call_on_device(f, x) ≈ f(x) broken = broken
    end
end

@testset "binary - $T" for T in float_types
    @testset "$f" for f in [
            atan,
            copysign,
            max,
            min,
            hypot,
            (^),
        ]
        x = rand(T)
        y = rand(T)
        broken = ispocl && T == Float16 && f == atan
        @test call_on_device(f, x, y) ≈ f(x, y) broken = broken
    end
end

@testset "ternary - $T" for T in float_types
    @testset "$f" for f in [
            fma,
        ]
        x = rand(T)
        y = rand(T)
        z = rand(T)
        @test call_on_device(f, x, y, z) ≈ f(x, y, z)
    end
end

@testset "OpenCL-specific unary - $T" for T in float_types
    @testset "$f" for f in [
            OpenCL.acospi,
            OpenCL.asinpi,
            OpenCL.atanpi,
            OpenCL.logb,
            OpenCL.rint,
            OpenCL.rsqrt,
        ]
        x = rand(T)
        broken = ispocl && T == Float16 && !(f in [OpenCL.rint, OpenCL.rsqrt])
        @test call_on_device(f, x) isa Real broken = broken  # Just check it doesn't error
    end
    broken = ispocl && T == Float16
    @test call_on_device(OpenCL.ilogb, T(8.0)) isa Int32 broken = broken
    @test call_on_device(OpenCL.nan, Base.uinttype(T)(0)) isa T
end

@testset "OpenCL-specific binary - $T" for T in float_types
    @testset "$f" for f in [
            OpenCL.atanpi,
            OpenCL.dim,
            OpenCL.maxmag,
            OpenCL.minmag,
            OpenCL.nextafter,
            OpenCL.powr,
        ]
        x = rand(T)
        y = rand(T)
        broken = ispocl && T == Float16 && !(f in [OpenCL.maxmag, OpenCL.minmag])
        @test call_on_device(f, x, y) isa Real broken = broken  # Just check it doesn't error
    end
    broken = ispocl && T == Float16
    @test call_on_device(OpenCL.rootn, T(8.0), Int32(3)) ≈ T(2.0) broken = broken
end

@testset "OpenCL-specific ternary - $T" for T in float_types
    x = rand(T)
    y = rand(T)
    z = rand(T)
    @test call_on_device(OpenCL.mad, x, y, z) ≈ x * y + z
end

@testset "SIMD - $N x $T" for N in simd_ns, T in float_types
    v = Vec{N, T}(ntuple(_ -> rand(T), N))

    # unary ops: sin, cos, sqrt
    a = call_on_device(sin, v)
    @test all(a[i] ≈ sin(v[i]) for i in 1:N)

    b = call_on_device(cos, v)
    @test all(b[i] ≈ cos(v[i]) for i in 1:N)

    c = call_on_device(sqrt, v)
    @test all(c[i] ≈ sqrt(v[i]) for i in 1:N)

    # binary ops: max, hypot
    w = Vec{N, T}(ntuple(_ -> rand(T), N))
    d = call_on_device(max, v, w)
    @test all(d[i] == max(v[i], w[i]) for i in 1:N)

    broken = ispocl && T == Float16
    if !broken
        h = call_on_device(hypot, v, w)
        @test all(h[i] ≈ hypot(v[i], w[i]) for i in 1:N)
    end

    # ternary op: fma
    x = Vec{N, T}(ntuple(_ -> rand(T), N))
    e = call_on_device(fma, v, w, x)
    @test all(e[i] ≈ fma(v[i], w[i], x[i]) for i in 1:N)

    # special cases: ilogb, ldexp, ^ with Int32, rootn
    v_pos = Vec{N, T}(ntuple(_ -> rand(T) + T(1), N))
    @test call_on_device(OpenCL.ilogb, v_pos) isa Vec{N, Int32} broken = broken

    k = Vec{N, Int32}(ntuple(_ -> rand(Int32.(-5:5)), N))
    @test let
        ldexp_result = call_on_device(ldexp, v_pos, k)
        all(ldexp_result[i] ≈ ldexp(v_pos[i], k[i]) for i in 1:N)
    end broken = broken

    base = Vec{N, T}(ntuple(_ -> rand(T) + T(0.5), N))
    exp_int = Vec{N, Int32}(ntuple(_ -> rand(Int32.(0:3)), N))
    pow_result = call_on_device(^, base, exp_int)
    @test all(pow_result[i] ≈ base[i] ^ exp_int[i] for i in 1:N)

    rootn_base = Vec{N, T}(ntuple(_ -> rand(T) * T(10) + T(1), N))
    rootn_n = Vec{N, Int32}(ntuple(_ -> rand(Int32.(2:4)), N))
    @test call_on_device(OpenCL.rootn, rootn_base, rootn_n) isa Vec{N, T} broken = broken

    # special cases: nan
    nan_code = Vec{N, Base.uinttype(T)}(ntuple(_ -> rand(Base.uinttype(T)), N))
    nan_result = call_on_device(OpenCL.nan, nan_code)
    @test all(isnan(nan_result[i]) for i in 1:N)
end

end

end
