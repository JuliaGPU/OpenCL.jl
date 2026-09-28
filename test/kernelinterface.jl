import KernelInterface
import KernelInterface as KI
using OpenCL.OpenCLInterface

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(OpenCLInterface.OpenCLBackend(), CLArray)

function ki_fill_kernel(a, val)
    i = KI.get_global_id().x
    if i <= length(a)
        @inbounds a[i] = val
    end
    return
end

@testset "backend platform" begin
    backend = OpenCLInterface.OpenCLBackend()
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
            other = OpenCLInterface.OpenCLBackend(; platform = first(others))
            @test KI.device(other) == 1
            b = KI.zeros(other, Int32, 4)
            @test cl.platform() == first(others)
            @test KI.get_backend(b) == other
            @test KI.device(other, b) == KI.device(other)
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

function ki_slow_kernel(a, iters)
    acc = UInt32(KI.get_global_id().x)
    for k in UInt32(1):iters
        acc = acc * 0x0019660d + k
    end
    @inbounds a[1] = acc
    return
end

@testset "cooperative synchronize" begin
    backend = OpenCLInterface.OpenCLBackend()
    a = KI.zeros(backend, UInt32, 1)
    KI.@launch backend ki_slow_kernel(a, UInt32(1))
    KI.synchronize(backend)

    # another task on this thread gets to run while `synchronize` waits for the device
    done = Ref(false)
    ticks = Ref(0)
    task = @async while !done[]
        ticks[] += 1
        yield()
    end
    KI.@launch backend ki_slow_kernel(a, UInt32(2)^24)
    KI.synchronize(backend)
    during = ticks[]
    done[] = true
    wait(task)
    @test during > 0
end
