using KernelAbstractions
using Atomix: Atomix

@testset "spirv_extensions" begin

@testset "bitreverse KernelAbstractions kernel" begin
    @kernel function bitreverse_ka!(out, inp)
        i = @index(Global)
        @inbounds out[i] = bitreverse(inp[i])
    end

    @testset "$T" for T in [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64]
        N = 64
        inp = CLArray(rand(T, N))
        out = similar(inp)
        bitreverse_ka!(OpenCLBackend())(out, inp; ndrange=N)
        synchronize(OpenCLBackend())

        @test Array(out) == bitreverse.(Array(inp))
    end
end

@testset "atomic float accumulation without manual extensions" begin
    # The auto-spirv-extensions feature detects cl_ext_float_atomics and enables
    # SPV_EXT_shader_atomic_float_add automatically. Previously this required
    # passing extensions=["SPV_EXT_shader_atomic_float_add"] to @opencl manually.
    #
    # We test with a concurrent accumulation pattern where multiple work-items
    # write to the same output locations. Without atomics this would race;
    # with Atomix.@atomic (which emits OpAtomicFAddEXT) the result must be exact.
    if "cl_ext_float_atomics" in cl.device().extensions
        @kernel function atomic_accum_kernel!(out, arr)
            i, j = @index(Global, NTuple)
            for k in 1:size(out, 1)
                Atomix.@atomic out[k, i] += arr[i, j]
            end
        end

        @testset "$T" for T in [Float32, Float64]
            if T == Float64 && !("cl_khr_fp64" in cl.device().extensions)
                continue
            end

            M, N = 32, 64
            img = zeros(T, M, N)
            img[5:15, 5:15] .= one(T)
            img[20:30, 20:30] .= T(2)

            cl_img = CLArray(img)
            out = KernelAbstractions.zeros(OpenCLBackend(), T, M, N)
            atomic_accum_kernel!(OpenCLBackend())(out, cl_img; ndrange=(M, N))
            synchronize(OpenCLBackend())

            # Each out[k, i] = sum(img[i, :]) — accumulate row i across all columns
            out_host = Array(out)
            expected = zeros(T, M, N)
            for i in 1:M
                expected[:, i] .= sum(img[i, :])
            end
            @test out_host ≈ expected
        end
    end
end

end
