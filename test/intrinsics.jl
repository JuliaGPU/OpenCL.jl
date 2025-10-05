function call_on_device(f, args...)
    function kernel(res, f, args...)
        res[] = f(args...)
        return
    end
    res = CLArray{typeof(f(args...)), 0}(undef)
    @opencl kernel(res, f, args...)
    return OpenCL.@allowscalar res[]
end

const float_types = filter(x -> x <: Base.IEEEFloat, GPUArraysTestSuite.supported_eltypes(CLArray))

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
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_all_svm_devices)
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
@on_device atomic_work_item_fence(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_order_acq_rel, OpenCL.memory_scope_all_svm_devices)
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
        if T == Float16 && f in [acosh, asinh, atanh, cbrt, cospi, expm1, log1p, sinpi, tanpi]
            @test_broken call_on_device(f, x) ≈ f(x)
        else
            @test call_on_device(f, x) ≈ f(x)
        end
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
        if T == Float16 && f == atan
            @test_broken call_on_device(f, x, y) ≈ f(x, y)
        else
            @test call_on_device(f, x, y) ≈ f(x, y)
        end
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

@testset "OpenCL-specific unary" begin
    @on_device OpenCL.acospi(0.5f0)
    @on_device OpenCL.asinpi(0.5f0)
    @on_device OpenCL.atanpi(1.0f0)
    @on_device OpenCL.logb(2.0f0)
    @on_device OpenCL.rint(1.5f0)
    @on_device OpenCL.rsqrt(4.0f0)
    @on_device OpenCL.ilogb(8.0f0)
    @on_device OpenCL.nan(UInt32(0))
end

@testset "OpenCL-specific binary" begin
    @on_device OpenCL.atanpi(1.0f0, 1.0f0)
    @on_device OpenCL.dim(5.0f0, 3.0f0)
    @on_device OpenCL.maxmag(3.0f0, -4.0f0)
    @on_device OpenCL.minmag(3.0f0, -4.0f0)
    @on_device OpenCL.nextafter(1.0f0, 2.0f0)
    @on_device OpenCL.powr(2.0f0, 3.0f0)
    @on_device OpenCL.rootn(8.0f0, Int32(3))
end

@testset "OpenCL-specific ternary" begin
    @on_device OpenCL.mad(2.0f0, 3.0f0, 4.0f0)
end

end

end
