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

@testset "@Const" begin
    @kernel function const_copy_kernel(A, @Const(B))
        I = @index(Global)
        @inbounds A[I] = B[I]
    end

    @kernel function const_copy_kernel_2d(A, @Const(B))
        i, j = @index(Global, NTuple)
        @inbounds A[i, j] = B[i, j]
    end

    backend = OpenCL.OpenCLBackend()

    # a constified argument must still read back the values it was given,
    # both linearly and as an ND index
    A = KernelAbstractions.zeros(backend, Float32, 1024)
    B = KernelAbstractions.ones(backend, Float32, 1024)
    ir = sprint() do io
        @device_code_llvm io=io raw=true begin
            const_copy_kernel(backend, 8)(A, B, ndrange=length(A))
            KernelAbstractions.synchronize(backend)
        end
    end
    @test Array(A) == fill(1.0f0, 1024)
    # loads from a constified argument are marked as invariant
    @test occursin("!invariant.load", ir)

    A = KernelAbstractions.zeros(backend, Float32, 32, 32)
    B = KernelAbstractions.ones(backend, Float32, 32, 32)
    const_copy_kernel_2d(backend, (8, 8))(A, B, ndrange=size(A))
    KernelAbstractions.synchronize(backend)
    @test Array(A) == fill(1.0f0, 32, 32)
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

# the others run kernels on KernelAbstractions' PoCL-based CPU back end
skip_tests=Set([
    "sparse",
    "CPU synchronization",
    "fallback test: callable types",
])
KATestSuite.testsuite(OpenCLBackend, "OpenCL", OpenCL, CLArray, CLDeviceArray; skip_tests)

@kernel function store_global_linear!(A)
    I = @index(Global, Linear)
    @inbounds A[I] = I
end

@kernel function store_last_index!(A)
    I = @index(Global, Linear)
    if I == prod(@ndrange())
        @inbounds A[1] = I
        @inbounds A[2] = @index(Global, Cartesian)[2]
    end
end

@testset "launch configuration" begin
    backend = OpenCLBackend()
    function select(kernel, ndrange, workgroupsize=nothing)
        ndrange, workgroupsize, iterspace, _ = KernelAbstractions.launch_config(kernel, ndrange, workgroupsize)
        KernelAbstractions.select_launch(kernel, workgroupsize, iterspace)
    end

    # kernels are launched on an N-d grid, computing indices in 32 bits
    kernel = store_global_linear!(backend)
    @test select(kernel, (64, 32, 16)) === KernelAbstractions.NDLaunch{Int32}()
    @test select(kernel, (4, 4, 4, 4)) === KernelAbstractions.LinearLaunch{Int32}()

    # which doesn't need divisions to compute the index of a dynamic N-d range
    A = OpenCL.zeros(Int, 64, 32, 16)
    ir = sprint(io -> @device_code_llvm io=io kernel(A; ndrange=size(A)))
    @test !occursin(r"\b[su](div|rem) ", ir)
    @test Array(A) == LinearIndices(A)

    # iteration spaces that don't fit 32 bits use 64-bit indices
    kernel = store_last_index!(backend)
    A = OpenCL.zeros(Int, 2)
    for (dims, launch) in (((2^16 + 1, 2^15), KernelAbstractions.NDLaunch{Int}()),
                           ((2^11 + 1, 2^10, 2^10, 1), KernelAbstractions.LinearLaunch{Int}()))
        @test select(kernel, dims) === launch
        kernel(A; ndrange=dims)
        @test Array(A) == [prod(dims), dims[2]]
    end
end
