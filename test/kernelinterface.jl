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

@testset "sub-groups" begin
    backend = OpenCLBackend()
    dev = cl.device()
    # kernels only execute with a fixed sub-group width if they can request one
    if cl.sub_groups_supported(dev) && "cl_intel_required_subgroup_size" in dev.extensions
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
    else
        @test !KI.supports_subgroups(backend)
    end
    @test !KI.supports_shuffle(backend, Complex{Float32})
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
