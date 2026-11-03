# On Julia 1.11+, `BFloat16` is `Core.BFloat16`, which Julia lowers to LLVM's `bfloat`:
# storing it needs a device that accepts `SPV_KHR_bfloat16`, and GPUCompiler widens its
# arithmetic to `float`. On Julia 1.10, BFloat16s.jl defines it as a 16-bit primitive type.

using BFloat16s

function bf16_copy(y, x)
    i = get_global_id()
    @inbounds y[i] = x[i]
    return
end
function bf16_ops(out, x, y)
    i = get_global_id()
    @inbounds begin
        a, b = x[i], y[i]
        out[1, i] = a * b + b
        out[2, i] = a / b
        out[3, i] = min(a, b)
        out[4, i] = max(a, b)
        out[5, i] = abs(a)
        out[6, i] = copysign(a, b)
        out[7, i] = ifelse(a < b, a, b)
        out[8, i] = sqrt(abs(a))
        out[9, i] = -a
    end
    return
end
function bf16_convert(y, x)
    i = get_global_id()
    @inbounds y[i] = convert(eltype(y), x[i])
    return
end

# the correctly rounded value, computed without the host's `BFloat16` conversions (on Julia
# 1.11, vectorized ones crash LLVM on AVX512-BF16 hosts)
bf16_nearest(x) = BFloat16(Float32(BigFloat(x; precision = 8)))

# finite values from bit patterns, rather than from rounding on the host (Julia 1.11's
# vectorized `Float32` to `BFloat16` conversion crashes LLVM on AVX512-BF16 hosts)
rand_bf16(n) = map(1:n) do _
    while true
        v = reinterpret(BFloat16, rand(UInt16))
        isfinite(v) && return v
    end
end

if !isdefined(Core, :BFloat16) || OpenCL.compiler_config(cl.device()).target.supports_bfloat16
    n = 1024
    hx, hy = rand_bf16(n), rand_bf16(n)
    hx[1:4] .= BFloat16.((-0.0, 0.0, 1.0, -2.0))
    hy[1:4] .= BFloat16.((0.0, -0.0, 0.0, 3.0))
    x, y = CLArray(hx), CLArray(hy)

    @testset "copy" begin
        out = similar(x)
        @opencl global_size = n bf16_copy(out, x)
        @test isequal(Array(out), hx)
    end

    @testset "arithmetic" begin
        out = CLArray{BFloat16}(undef, 9, n)
        @opencl global_size = n bf16_ops(out, x, y)
        o = Array(out)
        # (references in `Float32`: on Julia 1.11, the host miscompiles vectorized `BFloat16`
        # operations such as `copysign`)
        f32(op) = (xs...) -> BFloat16(op(Float32.(xs)...))
        # (multiplied and added separately, each rounded to bfloat)
        @test isequal(o[1, :], map((a, b) -> f32(+)(f32(*)(a, b), b), hx, hy))
        @test isequal(o[2, :], map(f32(/), hx, hy))
        @test isequal(o[3, :], map(f32(min), hx, hy))
        @test isequal(o[4, :], map(f32(max), hx, hy))
        @test isequal(o[5, :], map(f32(abs), hx))
        @test isequal(o[6, :], map(f32(copysign), hx, hy))
        @test isequal(o[7, :], map(f32((a, b) -> ifelse(a < b, a, b)), hx, hy))
        @test isequal(o[8, :], map(f32(a -> sqrt(abs(a))), hx))
        @test isequal(o[9, :], map(f32(-), hx))
    end

    # including inputs that a conversion through `Float32` would round wrongly
    conversions = (
        (Float32, Float32[1 + 2.0f0^-8 + 2.0f0^-20, floatmax(Float32)]),
        (Float64, [1 + 2.0^-8 + 2.0^-30, -(1 + 2.0^-8 + 2.0^-30), 1.0e39]),
        (Int32, Int32[2^30 + 2^22 + 1, typemin(Int32), typemax(Int32)]),
        (Int64, Int64[2^62 + 2^54 + 1, typemin(Int64), typemax(Int64)]),
        (UInt64, [UInt64(2)^63 + UInt64(2)^55 + 1, typemax(UInt64)]),
    )
    fp64 = "cl_khr_fp64" in cl.device().extensions
    @testset "$T to BFloat16" for (T, extra) in filter(c -> fp64 || c[1] != Float64, conversions)
        vals = T <: AbstractFloat ? T.(randn(n) .* 1000) : rand(T, n)
        vals[1:length(extra)] .= extra
        dst = CLArray{BFloat16}(undef, n)
        @opencl global_size = n bf16_convert(dst, CLArray(vals))
        @test isequal(Array(dst), map(bf16_nearest, vals))
    end

    @testset "BFloat16 to Float32" begin
        dst = CLArray{Float32}(undef, n)
        @opencl global_size = n bf16_convert(dst, x)
        @test isequal(Array(dst), map(Float32, hx))
    end
else
    # without `SPV_KHR_bfloat16` (or on the OpenCL C source path), compiling must fail
    # cleanly, rather than the driver aborting on a rejected extension
    x = CLArray(BFloat16.(Float32[1, 2, 3, 4]))
    @test_throws OpenCL.GPUCompiler.InvalidIRError @opencl global_size = 4 bf16_copy(similar(x), x)
end
