# KernelAbstractions has a testsuite that isn't part of the main package.
# Include it directly.

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
])
KATestSuite.testsuite(OpenCLBackend, "OpenCL", OpenCL, CLArray, CLDeviceArray; skip_tests)
