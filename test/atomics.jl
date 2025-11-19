using SPIRVIntrinsics: @builtin_ccall, @typed_ccall, LLVMPtr, known_intrinsics

skip_int64(T) = sizeof(T) == 8 && T <: Integer && !("cl_khr_int64_extended_atomics" in cl.device().extensions)
skip_float64(T) = T == Float64 && !("cl_khr_fp64" in cl.device().extensions)
integer_types = [Int32, UInt32, Int64, UInt64]
float_types = [Float32, Float64]
@testset "atomics" begin

function atomic_add_(counter, ::Val{T}) where T
    OpenCL.@atomic counter[] += one(T)
    return
end

@testset "atomic_add! ($T)" for T in vcat(integer_types, float_types)
    if skip_int64(T) || skip_float64(T)
        continue
    end
    @show T
    a = OpenCL.zeros(T)
    @opencl global_size=1000 atomic_add_(a, Val(T))
    @test OpenCL.@allowscalar a[] == T(1000)
end

function atomic_sub_(counter, ::Val{T}) where T
    OpenCL.@atomic counter[] -= one(T)
    return
end

@testset "atomic_sub! ($T)" for T in vcat(integer_types, float_types)
    if skip_int64(T) || skip_float64(T)
        continue
    end
    @show T
    a = T(1000.0)
    @opencl global_size=1000 atomic_sub_(a, Val(T))
    @test OpenCL.@allowscalar a[] == zero(T)
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
