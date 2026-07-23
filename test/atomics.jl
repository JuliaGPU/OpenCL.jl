using SPIRVIntrinsics: @builtin_ccall, @typed_ccall, LLVMPtr, known_intrinsics

# Define the types to test
integer_types = [Int32, UInt32, Int64, UInt64]
float_types = [Float16, Float32, Float64]
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

    # Skip Float16 if not supported
    if T == Float16 && !("cl_khr_fp16" in dev.extensions)
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


@testset "float atomics ($T)" for T in [Float16, Float32, Float64]
    if T == Float64 && !("cl_khr_fp64" in dev.extensions)
        continue
    end
    if T == Float64 && !("cl_khr_int64_base_atomics" in dev.extensions)
        continue
    end
    if T == Float16 && !("cl_khr_fp16" in dev.extensions)
        continue
    end

    a = OpenCL.zeros(T)
    @opencl global_size=1000 float_add_kernel(a, one(T))
    @test OpenCL.@allowscalar(a[]) == T(1000)

    b = OpenCL.fill(T(1000))
    @opencl global_size=1000 float_sub_kernel(b, one(T))
    @test OpenCL.@allowscalar(b[]) == T(0)

    # the native/fallback selection must fold at compile time, leaving only the path
    # matching the device's capabilities
    feature = T == Float32 ? :fp32_atomic_add :
              T == Float64 ? :fp64_atomic_add : :fp16_atomic_add
    native = OpenCL.feature_supported(dev, feature)
    if T == Float16 &&
       (OpenCL.program_backend() === :opencl || !("cl_khr_il_program" in dev.extensions))
        # the fp16 natives are masked on the OpenCL C source backend (see `_compiler_config`)
        native = false
    end
    ir = sprint() do io
        OpenCL.code_llvm(io, float_add_kernel, Tuple{CLDeviceArray{T, 0, AS.CrossWorkgroup}, T};
                         kernel=true, dump_module=true)
    end
    if native
        @test occursin("__spirv_AtomicFAddEXT", ir)
        @test !occursin("__spirv_AtomicCompareExchange", ir)
    else
        @test occursin("__spirv_AtomicCompareExchange", ir)
        @test !occursin("__spirv_AtomicFAddEXT", ir)
    end
end

@testset "atomic_add builtin ($T)" for T in [Float32, Float64]
    # Float64 requires cl_khr_fp64 extension
    if T == Float64 && !("cl_khr_fp64" in cl.device().extensions)
        continue
    end
if "cl_ext_float_atomics" in cl.device().extensions
    @eval function atomic_float_add(counter, val::$T)
        @builtin_ccall(
            "atomic_add", $T,
            (LLVMPtr{$T, AS.CrossWorkgroup}, $T),
            pointer(counter), val,
        )
        return
    end

    @testset "SPV_EXT_shader_atomic_float_add extension" begin
        a = OpenCL.zeros(T)
        @opencl global_size = 1000 extensions = ["SPV_EXT_shader_atomic_float_add"] atomic_float_add(a, one(T))
        @test OpenCL.@allowscalar a[] == T(1000.0)

        spv = sprint() do io
            OpenCL.code_native(io, atomic_float_add, Tuple{CLDeviceArray{T, 0, 1}, T}; extensions = ["SPV_EXT_shader_atomic_float_add"])
        end
        @test occursin("OpExtension \"SPV_EXT_shader_atomic_float_add\"", spv)
        @test occursin("OpAtomicFAddEXT", spv)
    end
end

end
end
