using SPIRVIntrinsics: @builtin_ccall, @typed_ccall, LLVMPtr, known_intrinsics
import SPIRVIntrinsics, SPIRV_LLVM_Translator_jll

# Define the types to test
integer_types = [Int32, UInt32, Int64, UInt64]
float_types = [Float32, Float64]
all_types = vcat(integer_types, float_types)

dev = OpenCL.cl.device()

# Arithmetic operations
function test_atomic_add(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] += one(T)
    return
end
function test_atomic_sub(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] -= one(T)
    return
end
# Bitwise operations
function test_atomic_and(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] &= ~(one(T) << (get_global_id() - 1))
    return
end
function test_atomic_or(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] |= one(T) << (get_global_id() - 1)
    return
end
function test_atomic_xor(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] ⊻= one(T) << ((get_global_id() - 1) % 32)
    return
end
# Min/max operations - use low-level API directly
function test_atomic_max(counter::AbstractArray{T}) where T
    OpenCL.atomic_max!(pointer(counter), T(get_global_id()))
    return
end
function test_atomic_min(counter::AbstractArray{T}) where T
    OpenCL.atomic_min!(pointer(counter), T(get_global_id()))
    return
end
# Exchange operation - use low-level API directly
function test_atomic_xchg(counter::AbstractArray{T}) where T
    OpenCL.atomic_xchg!(pointer(counter), one(T))
    return
end
# Compare-and-swap operation - use low-level API directly
function test_atomic_cas(counter::AbstractArray{T}) where T
    OpenCL.atomic_cmpxchg!(pointer(counter), zero(T), one(T))
    return
end
# Floating-point add/sub - use low-level API directly
function float_add_kernel(counter::AbstractArray{T}, val::T) where T
    OpenCL.atomic_add!(pointer(counter), val)
    return
end
function float_sub_kernel(counter::AbstractArray{T}, val::T) where T
    OpenCL.atomic_sub!(pointer(counter), val)
    return
end

# Define atomic operations to test
atomic_operations = [
    # op, init_val, expected_val
    (test_atomic_add, 0, 1000),
    (test_atomic_sub, 1000, 0),
    (test_atomic_and, typemax(UInt64), 0),
    (test_atomic_or, 0, typemax(UInt64)),
    (test_atomic_xor, 0, typemax(UInt32) << 8),
    (test_atomic_max, 0, 1000),
    (test_atomic_min, 1000, 1),
    (test_atomic_xchg, 0, 1),
    (test_atomic_cas, 0, 1),
]
@testset "atomics" begin
@testset "$kernel_func - $T" for (kernel_func, init_val, expected_val) in atomic_operations, T in all_types
    # Skip Int64/UInt64 if not supported
    if sizeof(T) == 8 && T <: Integer && !("cl_khr_int64_extended_atomics" in dev.extensions)
        continue
    end

    # Skip Float64 if not supported
    if T == Float64 && !("cl_khr_fp64" in dev.extensions)
        continue
    end

    # Float64 atomics may fall back to 64-bit cmpxchg
    if T == Float64 && !("cl_khr_int64_base_atomics" in dev.extensions)
        continue
    end

    # Bitwise operations (only valid for integers)
    if kernel_func in [test_atomic_and, test_atomic_or, test_atomic_xor] && T <: AbstractFloat
        continue
    end

    # Min/max on integers is only supported for 32-bit types; floats use the native
    # extension or the compare-and-swap fallback
    if kernel_func in [test_atomic_min, test_atomic_max] && !(T in [Int32, UInt32] || T <: AbstractFloat)
        continue
    end

    if T <: Integer
        init_val %= T
        expected_val %= T
    end

    a = OpenCL.fill(T(init_val))
    @opencl global_size=1000 kernel_func(a)
    result_val = OpenCL.@allowscalar a[]
    @test result_val === T(expected_val)
end


@testset "float atomics ($T)" for T in [Float32, Float64]
    if T == Float64 && !("cl_khr_fp64" in dev.extensions)
        continue
    end
    if T == Float64 && !("cl_khr_int64_base_atomics" in dev.extensions)
        continue
    end

    a = OpenCL.zeros(T)
    @opencl global_size=1000 float_add_kernel(a, one(T))
    @test OpenCL.@allowscalar(a[]) == T(1000)

    b = OpenCL.fill(T(1000))
    @opencl global_size=1000 float_sub_kernel(b, one(T))
    @test OpenCL.@allowscalar(b[]) == T(0)
end


# An atomic on global memory needs device scope to be atomic with respect to other
# work-groups; one on local memory only needs work-group scope.
function atomic_scope_kernel(op, a::AbstractArray{T}, val::T) where T
    op(pointer(a), val)
    return
end
function atomic_scope_kernel_local(op, a::AbstractArray{T}, val::T) where T
    s = CLLocalArray(T, (1,))
    op(pointer(s), val)
    @inbounds a[1] = s[1]
    return
end

# the Scope operand of every atomic instruction in a SPIR-V disassembly
function atomic_scopes(asm)
    constants = Dict(m[1] => parse(Int, m[2]) for m in
                     eachmatch(r"(%\S+) = OpConstant %\S+ (\d+)", asm))
    [constants[m[2]] for m in eachmatch(r"= (OpAtomic\w+) %\S+ %\S+ (%\S+)", asm)]
end

@testset "atomic scopes ($backend)" for backend in (:llvm, :khronos)
    ops = [
        (OpenCL.atomic_add!,                        UInt32,  "OpAtomicIAdd"),
        (OpenCL.atomic_xchg!,                       UInt32,  "OpAtomicExchange"),
        ((p, v) -> OpenCL.atomic_cmpxchg!(p, v, v), UInt32,  "OpAtomicCompareExchange"),
        (OpenCL.atomic_add!,                        Float32, "OpAtomicCompareExchange"),
    ]
    @testset "$inst ($T)" for (op, T, inst) in ops
        for (kernel, scope) in ((atomic_scope_kernel, 1),          # Scope.Device
                                (atomic_scope_kernel_local, 2))    # Scope.Workgroup
            asm = sprint() do io
                OpenCL.code_native(io, kernel,
                                   Tuple{typeof(op), CLDeviceArray{T, 1, AS.CrossWorkgroup}, T};
                                   kernel=true, backend)
            end
            @test occursin(inst, asm)
            scopes = atomic_scopes(asm)
            @test !isempty(scopes) && all(==(scope), scopes)
        end
    end
end

end


# `@atomic` returns the old value when it uses an atomic operation, and the new one when it
# falls back to a compare-and-swap loop.
function atomic_arrayset_kernel(a, out, val)
    @inbounds out[1] = OpenCL.@atomic a[1] += val
    @inbounds out[2] = OpenCL.@atomic a[2] *= val
    return
end
@testset "@atomic ($T)" for T in [Int32, Float32]
    cases = T <: Integer ? [(T(3), T(2)), (T(-3), T(2))] :
                           [(T(3), T(2)), (-zero(T), T(2)), (T(NaN), T(2)), (T(Inf), zero(T))]
    for (old, val) in cases
        a = CLArray([old, old])
        out = CLArray(zeros(T, 2))
        @opencl global_size=1 atomic_arrayset_kernel(a, out, val)
        @test isequal(Array(a), [old + val, old * val])
        @test isequal(Array(out), [old, old * val])
    end
end
