using SPIRVIntrinsics: @builtin_ccall, @typed_ccall, LLVMPtr, known_intrinsics

# Define the types to test
integer_types = [Int32, UInt32, Int64, UInt64]
float_types = [Float32, Float64]
all_types = vcat(integer_types, float_types)

dev = OpenCL.cl.device()
# Define atomic operations to test, with init value and expected value
atomic_operations = [
    (:atomic_add!, 0, 1),
    (:atomic_sub!, 1, 0),
    (:atomic_and!, 3, 1),
    (:atomic_or!, 2, 3),
    (:atomic_xor!, 3, 2),
    (:atomic_max!, 0, 1),
    (:atomic_min!, 2, 1),
    (:atomic_xchg!, 0, 1),
    (:atomic_cas!, 0, 1),
]
@testset "atomics" begin
for (op, init_val, expected_val) in atomic_operations
    for T in all_types
        # Skip Int64/UInt64 if not supported
        if sizeof(T) == 8 && T <: Integer && !("cl_khr_int64_extended_atomics" in dev.extensions)
            continue
        end

        # Skip Float64 if not supported
        if T == Float64 && !("cl_khr_fp64" in dev.extensions)
            continue
        end

        # Bitwise operations (only valid for integers)
        if op in [:atomic_and!, :atomic_or!, :atomic_xor!] && T <: AbstractFloat
            continue
        end

        # Min/max operations (only supported for 32-bit integers in OpenCL)
        if op in [:atomic_min!, :atomic_max!] && !(T in [Int32, UInt32])
            continue
        end

        test_name = Symbol("test_", op, "_", T)

        if op in [:atomic_add!, :atomic_sub!]
            # Arithmetic operations
            if op == :atomic_add!
                @eval function $test_name(counter)
                    OpenCL.@atomic counter[] += one($T)
                    return
                end
            else
                @eval function $test_name(counter)
                    OpenCL.@atomic counter[] -= one($T)
                    return
                end
            end
        elseif op in [:atomic_and!, :atomic_or!, :atomic_xor!]
            # Bitwise operations
            if op == :atomic_and!
                @eval function $test_name(counter)
                    OpenCL.@atomic counter[] &= one($T)
                    return
                end
            elseif op == :atomic_or!
                @eval function $test_name(counter)
                    OpenCL.@atomic counter[] |= one($T)
                    return
                end
            else # xor
                @eval function $test_name(counter)
                    OpenCL.@atomic counter[] ⊻= one($T)
                    return
                end
            end
        elseif op in [:atomic_max!, :atomic_min!]
            # Min/max operations - use low-level API directly
            if op == :atomic_max!
                @eval function $test_name(counter)
                    ptr = OpenCL.pointer(counter, 1)
                    OpenCL.atomic_max!(ptr, one($T))
                    return
                end
            else
                @eval function $test_name(counter)
                    ptr = OpenCL.pointer(counter, 1)
                    OpenCL.atomic_min!(ptr, one($T))
                    return
                end
            end
        elseif op == :atomic_xchg!
            # Exchange operation - use low-level API directly
            @eval function $test_name(counter)
                ptr = OpenCL.pointer(counter, 1)
                OpenCL.atomic_xchg!(ptr, one($T))
                return
            end
        elseif op == :atomic_cas!
            # CAS operation - use low-level API directly (it's called atomic_cmpxchg!)
            @eval function $test_name(counter)
                ptr = OpenCL.pointer(counter, 1)
                OpenCL.atomic_cmpxchg!(ptr, $T(0), one($T))
                return
            end
        else
            error("Unknown operation: $op")
        end


        # Try to compile the kernel - this is the key test
        a = OpenCL.zeros(T)
        OpenCL.fill!(a, init_val)
        kernel_func = @eval $test_name
        OpenCL.@opencl kernel_func(a)
        result_val = Array(a)[1]
        @test result_val == expected_val
    end
end


@testset "atomic_add! ($T)" for T in [Float32, Float64]
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
