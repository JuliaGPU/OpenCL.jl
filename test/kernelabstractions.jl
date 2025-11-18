# KernelAbstractions has a testsuite that isn't part of the main package.
# Include it directly.

using KernelAbstractions, StaticArrays

@testset "#422: SMatrix return" begin
    @kernel function smatrix_return_kernel(out)
        i = @index(Global)
        A = SMatrix{1,1,Float32}(3.0f0)
        B = SMatrix{1,1,Float32}(2.0f0)
        out[i] = (A * B)[1]
    end

    backend = OpenCL.OpenCLBackend()
    out = KernelAbstractions.zeros(backend, Float32, 4)
    kernel = smatrix_return_kernel(backend, 4)
    kernel(out, ndrange=4)
    KernelAbstractions.synchronize(backend)
    @test Array(out) == fill(6.0f0, 4)
end

const KATestSuite = let
    mod = @eval module $(gensym())
        using ..Test
        import KernelAbstractions
        kernelabstractions = pathof(KernelAbstractions)
        kernelabstractions_root = dirname(dirname(kernelabstractions))
        include(joinpath(kernelabstractions_root, "test", "testsuite.jl"))
    end
    mod.Testsuite
end

skip_tests=Set([
    "sparse",
    "Convert", # Need to opt out of i128
    "CPU synchronization",
    "fallback test: callable types"
])
KATestSuite.testsuite(OpenCLBackend, "OpenCL", OpenCL, CLArray, CLDeviceArray; skip_tests)
