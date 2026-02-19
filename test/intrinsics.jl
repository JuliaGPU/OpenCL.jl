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
# XXX: Why does pocl on windows not support vectors of size 2, 8, 16?
const simd_ns = (Sys.iswindows() && ispocl) ? [3, 4] : [2, 3, 4, 8, 16]

@testset "intrinsics" begin

@testset "barrier" begin

# work-group
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

# sub-group
@on_device sub_group_barrier(OpenCL.LOCAL_MEM_FENCE)
@on_device sub_group_barrier(OpenCL.GLOBAL_MEM_FENCE)
@on_device sub_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
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

if cl.sub_groups_supported(cl.device())

struct SubgroupData
    sub_group_size::UInt32
    max_sub_group_size::UInt32
    num_sub_groups::UInt32
    sub_group_id::UInt32
    sub_group_local_id::UInt32
end
function test_subgroup_kernel(results)
    i = get_global_id(1)

    if i <= length(results)
        @inbounds results[i] = SubgroupData(
            get_sub_group_size(),
            get_max_sub_group_size(),
            get_num_sub_groups(),
            get_sub_group_id(),
            get_sub_group_local_id()
        )
    end
    return
end

@testset "Sub-groups" begin
    sg_size = cl.sub_group_size(cl.device())

    @testset "Indexing intrinsics" begin
        # Test with small kernel
        sg_n = 2
        local_size = sg_size * sg_n
        numworkgroups = 2
        N = local_size * numworkgroups

        results = CLVector{SubgroupData}(undef, N)
        kernel = @opencl launch = false test_subgroup_kernel(results)

        kernel(results; local_size, global_size=N)

        host_results = Array(results)

        # Verify results make sense
        for (i, sg_data) in enumerate(host_results)
            @test sg_data.sub_group_size == sg_size
            @test sg_data.max_sub_group_size == sg_size
            @test sg_data.num_sub_groups == sg_n

            # Group ID should be 1-based
            expected_sub_group = div(((i - 1) % local_size), sg_size) + 1
            @test sg_data.sub_group_id == expected_sub_group

            # Local ID should be 1-based within group
            expected_sg_local = ((i - 1) % sg_size) + 1
            @test sg_data.sub_group_local_id == expected_sg_local
        end
    end

    @testset "shuffle idx" begin
        function shfl_idx_kernel(d)
            i = get_sub_group_local_id()
            j = get_sub_group_size() - i + 0x1

            d[i] = sub_group_shuffle(d[i], j)

            return
        end

        @testset for T in cl.sub_group_shuffle_supported_types(cl.device())
            a = rand(T, sg_size)
            d_a = CLArray(a)
            @opencl local_size = sg_size global_size = sg_size shfl_idx_kernel(d_a)
            @test Array(d_a) == reverse(a)
        end
    end
    @testset "shuffle xor" begin
        function shfl_xor_kernel(in)
            i = get_sub_group_local_id()

            new_val = sub_group_shuffle_xor(in[i], 1)

            in[i] = new_val
            return
        end

        # tests that each pair of values a get swapped using sub_group_shuffle_xor
        @testset for T in cl.sub_group_shuffle_supported_types(cl.device())
            in = rand(T, sg_size)
            idxs = xor.(0:(sg_size - 1), 1) .+ 1
            d_in = CLArray(in)
            @opencl local_size = sg_size global_size = sg_size shfl_xor_kernel(d_in)
            @test Array(d_in) == in[idxs]
        end
    end
end
end # if cl.sub_groups_supported(cl.device())

@testset "SIMD - $N x $T" for N in simd_ns, T in float_types
    # codegen emits i48 here, which SPIR-V doesn't support
    # XXX: fix upstream?
    T == Float16 && N == 3 && continue

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
    @test let
        pow_result = call_on_device(^, base, exp_int)
        all(pow_result[i] ≈ base[i] ^ exp_int[i] for i in 1:N)
    end broken = broken

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
