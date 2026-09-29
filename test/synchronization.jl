function slow_fill(a, v)
    i = get_global_id()
    # enough work that the kernel is unlikely to have finished when the launch returns
    acc = 0f0
    for j in 1:10_000
        acc += sin(Float32(j))
    end
    @inbounds a[i] = v + 0f0 * acc
    return
end

# a marker completes once all previously submitted work on the current queue has
marker() = cl.enqueue_marker_with_wait_list(cl.AbstractEvent[])

throwing_kernel(a) = (a[length(a) + 1] = 1f0; return)

@testset "synchronize" begin
    a = OpenCL.zeros(Float32, 1024)
    @opencl global_size=length(a) slow_fill(a, 1f0)
    evt = marker()
    @test OpenCL.synchronize() === nothing
    @test evt.status == :complete
    @test all(==(1f0), Array(a))

    # an explicit queue
    q = cl.CmdQueue()
    evt = cl.queue!(q) do
        @opencl global_size=length(a) slow_fill(a, 2f0)
        marker()
    end
    @test OpenCL.synchronize(q) === nothing
    @test evt.status == :complete
    @test all(==(2f0), Array(a))

    # device-side exceptions are reported
    @opencl throwing_kernel(a)
    @test_throws OpenCL.KernelException OpenCL.synchronize()
end

@testset "@sync" begin
    a = OpenCL.zeros(Float32, 1024)
    evt = OpenCL.@sync begin
        @opencl global_size=length(a) slow_fill(a, 1f0)
        marker()
    end
    @test evt.status == :complete

    # returns the value of the expression, evaluating it only once
    evaluations = Ref(0)
    b = OpenCL.@sync begin
        evaluations[] += 1
        OpenCL.zeros(Float32, 4) .+ 1
    end
    @test evaluations[] == 1
    @test Array(b) == ones(Float32, 4)

    # doesn't clobber the caller's variables
    ret = 1
    x = OpenCL.@sync 2
    @test ret == 1
    @test x == 2

    @test_throws OpenCL.KernelException OpenCL.@sync @opencl throwing_kernel(a)
end

@testset "public API" begin
    if VERSION >= v"1.11.0-DEV.469"
        @test Base.ispublic(OpenCL, :synchronize)
        @test Base.ispublic(OpenCL, Symbol("@sync"))
    end
    @test !Base.isexported(OpenCL, :synchronize)
    @test !Base.isexported(OpenCL, Symbol("@sync"))
end
