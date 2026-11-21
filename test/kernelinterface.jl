import KernelInterface
import KernelInterface as KI
import Adapt

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(OpenCLBackend(), CLArray)

function ki_fill_kernel(a, val)
    i = KI.get_global_id().x
    if i <= length(a)
        @inbounds a[i] = val
    end
    return
end

@testset "backend platform" begin
    backend = OpenCLBackend()
    a = KI.zeros(backend, Int32, 4)
    @test KI.get_backend(a) == backend

    # a kernel belongs to the device it was compiled for
    kernel = KI.@launch backend launch=false ki_fill_kernel(a, Int32(1))
    cl.context!(cl.Context(cl.device())) do
        @test_throws ArgumentError kernel(a, Int32(1); ndrange = 4)
    end
    kernel(a, Int32(1); ndrange = 4)
    @test Array(a) == ones(Int32, 4)

    # a backend for another platform activates it
    others = filter(!=(cl.platform()), cl.platforms())
    if !isempty(others)
        platform, device = cl.platform(), cl.device()
        try
            other = OpenCLBackend(; platform = first(others))
            @test KI.device(other) == 1
            b = KI.zeros(other, Int32, 4)
            @test cl.platform() == first(others)
            @test KI.get_backend(b) == other
            @test KI.device(other, b) == KI.device(other)

            # so does moving arrays to it
            cl.platform!(platform)
            c = Adapt.adapt(other, Int32[1, 2])
            @test cl.platform() == first(others)
            @test KI.get_backend(c) == other
            KI.@launch other ndrange = 4 ki_fill_kernel(b, Int32(2))
            @test Array(b) == fill(Int32(2), 4)

            # arrays keep their platform
            cl.device!(device)
            @test KI.get_backend(b) == other
            @test KI.get_backend(a) == backend
        finally
            cl.platform!(platform)
            cl.device!(device)
        end
    end
end

struct KICapturedArray{A}
    array::A
end
Adapt.@adapt_structure KICapturedArray
function (f::KICapturedArray)()
    @inbounds f.array[1] = 7
    return
end

@testset "captured arrays" begin
    function captured_kernel()
        array = CLArray(Int32[0])
        kernel = KI.@launch OpenCLBackend() launch=false KICapturedArray(array)()
        return kernel, WeakRef(array)
    end

    # the kernel keeps the callable alive, and converts it again at launch, which makes the
    # launch's queue the owner of the arrays it captures
    kernel, owner = captured_kernel()
    GC.gc(true)
    @test owner.value !== nothing
    queue = cl.CmdQueue()
    cl.queue!(queue) do
        kernel()
        @test owner.value.data[].queue === queue
        cl.finish(queue)
    end
    @test Array(owner.value) == Int32[7]
end

@testset "launch keywords" begin
    a = KI.zeros(OpenCLBackend(), Int32, 4)
    kernel = KI.@launch OpenCLBackend() launch=false ki_fill_kernel(a, Int32(1))

    # OpenCL's launch options are passed on
    kernel(a, Int32(3); ndrange = 4, wait_on = cl.Event[])
    @test Array(a) == fill(Int32(3), 4)

    # but not ones that would override the launch geometry
    @test_throws ArgumentError kernel(a, Int32(1); ndrange = 4, global_size = 8)
    @test_throws ArgumentError kernel(a, Int32(1); ndrange = 4, local_size = 2)
end

# uses the last byte of the block
function ki_dynamic_localmem_kernel(out)
    n = KI.dynamic_localmemory_size()
    x = KI.dynamic_localmemory(UInt8, n)
    if n > 0
        @inbounds x[n] = 1
        @inbounds out[1] = n + x[n] - 1
    else
        @inbounds out[1] = 0
    end
    return
end

# dynamic and static local memory, next to the local memory of the RNG
function ki_dynamic_localmem_rng_kernel(out, ::Val{N}) where {N}
    n = KI.get_local_size().x
    i = KI.get_local_id().x
    I = KI.get_global_id().x
    x = KI.dynamic_localmemory(Float32, n)
    s = KI.localmemory(Int32, N)
    @inbounds x[i] = rand(Float32)
    @inbounds s[i] = i
    KI.barrier()
    j = n - i + 1
    @inbounds out[I, 1] = x[j]
    @inbounds out[I, 2] = s[j]
    @inbounds out[I, 3] = KI.dynamic_localmemory_size()
    return
end

function ki_static_localmem_kernel(out, ::Val{N}) where {N}
    s = KI.localmemory(Int32, N)
    i = KI.get_local_id().x
    @inbounds s[i] = i
    KI.barrier()
    @inbounds out[1] = s[1] + KI.dynamic_localmemory_size()
    return
end

@testset "dynamic local memory" begin
    backend = OpenCLBackend()
    @test KI.supports_dynamic_localmemory(backend)
    out = KI.zeros(backend, Int, 1)
    kernel = KI.@launch backend launch=false ki_dynamic_localmem_kernel(out)

    # kernels get the size they asked for, which isn't rounded up
    kernel(out; localmem = 24)
    @test Array(out) == [24]
    kernel(out; localmem = 5)
    @test Array(out) == [5]
    kernel(out; localmem = n -> 8n, workgroupsize = 5)
    @test Array(out) == [40]
    kernel(out)
    @test Array(out) == [0]

    # it is a launch keyword of OpenCL.jl's kernels
    kernel.kern(out; localmem = 12)
    @test Array(out) == [12]
    @test_throws ArgumentError kernel.kern(out; localmem = -1)
    @test_throws ArgumentError kernel.kern(out; localmem = Int(typemax(UInt32)) + 1)

    # the limit is the local memory of the device, less the bytes that align the block
    # and what the driver reports the kernel to use (NVIDIA's reports some for any kernel)
    limit = KI.max_dynamic_localmemory(kernel)
    @test 0 < limit <= min(cl.device().local_mem_size, typemax(UInt32)) -
                       OpenCL.dynamic_localmemory_alignment
    kernel(out; localmem = limit)
    @test Array(out) == [limit]
    @test_throws ArgumentError kernel(out; localmem = limit + 1)
    # and stays that after launches have sized the argument
    @test KI.max_dynamic_localmemory(KI.@launch backend launch=false ki_dynamic_localmem_kernel(out)) == limit

    # kernels that don't use the memory have no argument for it, even if they ask for
    # its size
    a = KI.zeros(backend, Int32, 4)
    fill_kernel = KI.@launch backend launch=false ki_fill_kernel(a, Int32(1))
    @test !fill_kernel.kern.dynamic_localmemory
    @test kernel.kern.dynamic_localmemory
    fill_kernel(a, Int32(6); ndrange = 4, localmem = 64)
    @test Array(a) == fill(Int32(6), 4)

    # static local memory and the local memory of the RNG count against the limit
    N = 16
    static_kernel = KI.@launch backend launch=false ki_static_localmem_kernel(out, Val(N))
    @test !static_kernel.kern.dynamic_localmemory
    @test KI.max_dynamic_localmemory(static_kernel) <= limit
    static_kernel(out, Val(N); workgroupsize = N, localmem = 32)
    @test Array(out) == [33]

    groups = 3
    rng_out = KI.zeros(backend, Float32, N * groups, 3)
    rng_kernel = KI.@launch backend launch=false ki_dynamic_localmem_rng_kernel(rng_out, Val(N))
    @test rng_kernel.kern.rng_state
    @test rng_kernel.kern.dynamic_localmemory
    @test KI.max_dynamic_localmemory(rng_kernel) < KI.max_dynamic_localmemory(static_kernel)
    for localmem in (N * sizeof(Float32), KI.max_dynamic_localmemory(rng_kernel))
        rng_kernel(rng_out, Val(N); numgroups = groups, workgroupsize = N, localmem)
        result = Array(rng_out)
        @test all(x -> 0 <= x < 1, result[:, 1])
        @test length(unique(result[:, 1])) > 1
        @test result[:, 2] == repeat(Float32.(N:-1:1), groups)
        @test all(==(localmem), result[:, 3])
    end
end

@testset "sub-groups" begin
    backend = OpenCLBackend()
    dev = cl.device()
    # kernels only execute with a fixed sub-group width if they can request one
    # (of at most 64 lanes, for the ballot)
    if cl.sub_groups_supported(dev) && "cl_intel_required_subgroup_size" in dev.extensions &&
       cl.sub_group_size(dev) <= 64
        @test KI.supports_subgroups(backend)
        @test KI.sub_group_size(backend) == cl.sub_group_size(dev)
        @test KI.supports_shuffle(backend, Int32) ==
              ("cl_khr_subgroup_shuffle" in dev.extensions)

        # which kernels can't opt out of
        a = KI.zeros(backend, Int32, 4)
        width = KI.sub_group_size(backend)
        KI.@launch backend launch=false sub_group_size=width ki_fill_kernel(a, Int32(1))
        @test_throws ArgumentError KI.@launch backend launch=false sub_group_size=nothing ki_fill_kernel(a, Int32(1))
        @test_throws ArgumentError KI.@launch backend launch=false sub_group_size=2width ki_fill_kernel(a, Int32(1))
        # PoCL executes sub-group operations as work-group barriers
        occursin("pocl", dev.platform.vendor) && @test !KI.supports_independent_subgroups(backend)
    else
        @test !KI.supports_subgroups(backend)
        @test !KI.supports_linear_subgroups(backend)
        @test !KI.supports_independent_subgroups(backend)
    end
    # structs are shuffled field by field
    @test KI.supports_shuffle(backend, Complex{Float32}) == KI.supports_shuffle(backend, Float32)
end

# the ballot of devices without `cl_khr_subgroup_ballot`
function reduction_ballot_kernel(out, pred)
    i = KI.get_local_id().x
    @inbounds out[i, 1] = OpenCL.OpenCLKernels.reduction_ballot(pred[i])
    @inbounds out[i, 2] = KI.sub_group_ballot(pred[i])
    return
end

@testset "reduction ballot" begin
    backend = OpenCLBackend()
    if KI.supports_subgroups(backend)
        n = 2 * KI.sub_group_size(backend) + 5
        pred = CLArray(rand(Bool, n))
        out = CLArray(zeros(UInt64, n, 2))
        kernel = KI.@launch backend launch=false reduction_ballot_kernel(out, pred)
        if n <= KI.max_work_group_size(kernel)
            kernel(out, pred; workgroupsize = n)
            out = Array(out)
            @test out[:, 1] == out[:, 2]
        end
    end
end

@testset "events" begin
    backend = OpenCLBackend()
    a = KI.zeros(backend, Int32, 4)
    KI.@launch backend ndrange = 4 ki_fill_kernel(a, Int32(5))
    event = KI.record_event(backend)
    @test event isa cl.Event

    # queues of another context can't wait for the event, so the host does
    cl.context!(cl.Context(cl.device())) do
        @test KI.wait_event(backend, event) === nothing
        @test event.status == :complete
    end
    @test Array(a) == fill(Int32(5), 4)
end

@testset "versioninfo" begin
    @test occursin("OpenCL.jl version", sprint(KI.versioninfo, OpenCLBackend()))
end
