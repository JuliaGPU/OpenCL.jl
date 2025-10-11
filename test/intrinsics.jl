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

end

end
