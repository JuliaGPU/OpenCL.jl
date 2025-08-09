using SPIRVIntrinsics: @builtin_ccall, @typed_ccall, LLVMPtr

@testset "atomics" begin

function atomic_count(counter)
    OpenCL.@atomic counter[] += 1
    return
end

@testset "atomic_add! ($T)" for T in [Int32, UInt32, Int64, UInt64]
    if sizeof(T) == 4 || "cl_khr_int64_extended_atomics" in cl.device().extensions
        a = OpenCL.zeros(T)
        @opencl global_size=1000 atomic_count(a)
        @test OpenCL.@allowscalar a[] == 1000
    end
end

    if "cl_ext_float_atomics" in cl.device().extensions
        function atomic_float_add(counter, val)
            @builtin_ccall(
                "atomic_add", Float32,
                (LLVMPtr{Float32, AS.CrossWorkgroup}, Float32),
                pointer(counter), val,
            )
            return
        end

        @testset "SPV_EXT_shader_atomic_float_add extension" begin
            a = OpenCL.zeros(Float32)
            @opencl global_size = 1000 extensions = ["SPV_EXT_shader_atomic_float_add"] atomic_float_add(a, 1.0f0)
            @test OpenCL.@allowscalar a[] == 1000.0f0

            spv = sprint() do io
                OpenCL.code_native(io, atomic_float_add, Tuple{CLDeviceArray{Float32, 0, 1}, Float32}; extensions = ["SPV_EXT_shader_atomic_float_add"])
            end
            @test occursin("OpExtension \"SPV_EXT_shader_atomic_float_add\"", spv)
            @test occursin("OpAtomicFAddEXT", spv)
        end
    end

end
