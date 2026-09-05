using KernelAbstractions

@testset "debug level" begin
    # `debug_level` selects how much exception-reporting code a kernel carries,
    # independent of the session's `-g` level
    kernel(a) = (a[2] = 1f0; return)
    a = OpenCL.zeros(Float32, 1)
    ir(dl) = sprint(io->(@device_code_llvm io=io @opencl launch=false debug_level=dl kernel(a)))
    @test !occursin("gpu_report_exception", ir(0))
    @test occursin("gpu_report_exception", ir(1))
    @test !occursin("gpu_report_exception_frame", ir(1))
    @test occursin("gpu_report_exception_frame", ir(2))
end

@testset "device-side exceptions" begin
    # a kernel that throws must not wedge the device: it should complete, and surface on
    # the host as a `KernelException` when synchronizing the queue it ran on.
    function throwing_kernel(a)
        a[2] = 1f0      # out-of-bounds store on a length-1 array
        return
    end
    function fill_one(a)
        a[get_global_id()] = 1f0
        return
    end

    a = OpenCL.zeros(Float32, 1)
    @opencl throwing_kernel(a)
    @test_throws OpenCL.KernelException cl.finish(cl.queue())
    # the mailbox is reset on read
    cl.finish(cl.queue())

    # A context shares one mailbox across queues. Synchronizing either queue must first finish
    # every queue that may still use it, and may therefore surface the other queue's exception.
    plain_throwing_kernel() = error("device failure")
    no_op_kernel() = return
    q1, q2 = cl.CmdQueue(), cl.CmdQueue()
    cl.queue!(q1) do
        @opencl plain_throwing_kernel()
    end
    cl.queue!(q2) do
        @opencl no_op_kernel()
    end
    @test_throws OpenCL.KernelException cl.finish(q2)
    cl.finish(q1)

    # Finalizers synchronize without throwing, but must leave the report available for the
    # next ordinary synchronization.
    @opencl throwing_kernel(a)
    cl.finish(cl.queue(); check_exceptions=false)
    @test_throws OpenCL.KernelException cl.finish(cl.queue())

    # an exception whose argument needs boxing (a runtime value) must not have its throw
    # path deleted by the device compiler (JuliaGPU/GPUCompiler.jl#919)
    function boxed_kernel(a)
        x = a[1]
        x == 0 && throw(DomainError(x))
        return
    end
    ir = sprint(io->(@device_code_llvm io=io @opencl launch=false boxed_kernel(a)))
    @test occursin("gpu_signal_exception", ir)
    @opencl boxed_kernel(a)
    @test_throws OpenCL.KernelException cl.finish(cl.queue())

    # a blocking copy of the kernel's output raises instead of returning garbage
    @opencl throwing_kernel(a)
    @test_throws OpenCL.KernelException Array(a)

    # the device stays usable afterwards
    b = OpenCL.zeros(Float32, 4)
    @opencl global_size=4 fill_one(b)
    cl.finish(cl.queue())
    @test Array(b) == ones(Float32, 4)

    # broadcast
    c = CLArray([1, 0, 1])
    d = 1 .÷ c
    @test_throws OpenCL.KernelException Array(d)

    # KernelAbstractions
    @kernel function ka_throwing_kernel(a)
        i = @index(Global)
        a[i+1] = 1f0
    end
    e = OpenCL.zeros(Float32, 1)
    ka_throwing_kernel(OpenCLBackend())(e; ndrange=1)
    @test_throws OpenCL.KernelException KernelAbstractions.synchronize(OpenCLBackend())

    # devices without host-accessible pointer memory put the mailbox in a mappable buffer
    # with a device address; exercise that path by swapping in such a mailbox
    if cl.bda_supported(cl.device())
        ctx = cl.context()
        mailbox = OpenCL.allocate_exception_mailbox(ctx; backends=(:buffer,))
        @test mailbox.mem isa cl.Buffer
        old = Base.@lock OpenCL.exception_mailboxes_lock begin
            old = OpenCL.exception_mailboxes[(ctx, cl.device())]
            OpenCL.exception_mailboxes[(ctx, cl.device())] = mailbox
            old
        end
        try
            @opencl throwing_kernel(a)
            @test_throws OpenCL.KernelException cl.finish(cl.queue())
            cl.finish(cl.queue())
            @opencl throwing_kernel(a)
            @test_throws OpenCL.KernelException Array(a)
        finally
            Base.@lock OpenCL.exception_mailboxes_lock OpenCL.exception_mailboxes[(ctx, cl.device())] = old
        end
    end
end

@testset "exception reporting per debug level" begin
    # `@opencl debug_level=` controls how much an exception reports, independent of the
    # session's `-g` (it's part of the compile cache key, resolved at codegen), so all three
    # tiers can be checked in one process regardless of how the suite was launched.
    bounds_oob(a) = (a[2] = 1f0; return)   # out-of-bounds on a length-1 array
    nonquirk(out, d) = (out[1] = 1 ÷ d[1]; return)   # divide-by-zero -> throw(DivideError())
    function throw_at(dl, kernel, args...; kwargs...)
        k = @opencl launch=false debug_level=dl kernel(args...)
        k(args...; kwargs...)
        try; cl.finish(cl.queue()); nothing; catch err; err; end
    end

    # -g0: only that an exception occurred; everything else stays at its sentinel
    exc0 = throw_at(0, bounds_oob, OpenCL.zeros(Float32, 1))
    @test exc0 isa OpenCL.KernelException
    @test isempty(exc0.name)
    @test isempty(exc0.reason)
    @test exc0.work_item == (0, 0, 0)
    @test isempty(exc0.backtrace)
    @test occursin("debug_level=2", sprint(showerror, exc0))

    # -g1: the faulting type and reason for quirk throws; no position (recording it costs
    # every kernel that can throw, even when nothing ever does; see `claim_output!`)
    exc1 = throw_at(1, bounds_oob, OpenCL.zeros(Float32, 1))
    @test exc1.name == "BoundsError"
    @test exc1.reason == "Out-of-bounds array access"
    @test exc1.work_item == (0, 0, 0)
    @test isempty(exc1.backtrace)
    @test occursin("A BoundsError was thrown", sprint(showerror, exc1))

    # a non-quirk throw records the compiler-deduced type name (here `jl_throw`'s generic
    # "exception") only at debug level >= 2: copying that runtime string takes a byte loop
    # on the unhappy path (see `report_exception`)
    excn = throw_at(1, nonquirk, OpenCL.zeros(Int32, 1), CLArray(Int32[0]))
    @test excn isa OpenCL.KernelException
    @test isempty(excn.name)
    excn2 = throw_at(2, nonquirk, OpenCL.zeros(Int32, 1), CLArray(Int32[0]))
    @test excn2.name == "exception"

    # -g2: adds the faulting position and the device stack trace
    exc2 = throw_at(2, bounds_oob, OpenCL.zeros(Float32, 1))
    @test exc2.name == "BoundsError"
    @test exc2.work_item == (1, 1, 1)
    @test exc2.work_group == (1, 1, 1)
    @test !isempty(exc2.backtrace)
    @test any(frame -> occursin("throw_boundserror", frame[1]), exc2.backtrace)
    @test occursin("by work-item 1×1×1 in work-group 1×1×1", sprint(showerror, exc2))

    # the position identifies the faulting work-item, not the first one
    function throw_at_three(a)
        if get_local_id() == 3
            a[2] = 1f0
        end
        return
    end
    exc3 = throw_at(2, throw_at_three, OpenCL.zeros(Float32, 1);
                    global_size=8, local_size=4)
    @test exc3.work_item == (3, 1, 1)
    @test exc3.work_group in ((1, 1, 1), (2, 1, 1))

    # quirks that construct their exception without a `throw_*` helper are overridden to
    # report through the mailbox as well, rather than boxing on the throwing branch
    bool_kernel(a) = (a[1] = Bool(a[1]); return)
    excb = throw_at(1, bool_kernel, CLArray(Float32[2]))
    @test excb.name == "InexactError"
    exponent_kernel(a) = (a[1] = exponent(a[1]); return)
    exce = throw_at(1, exponent_kernel, CLArray(Float32[0]))
    @test exce.name == "DomainError"
end

@testset "exception report ownership" begin
    function first_failure()
        OpenCL.@gputhrow "FirstError" "first kernel"
        return
    end
    function second_failure()
        OpenCL.@gputhrow "SecondError" "second kernel"
        return
    end
    # Compile before submitting either kernel so compilation cannot trigger a finalizer
    # that synchronizes in between. The two launches have identical work-item coordinates.
    first = @opencl launch=false debug_level=2 first_failure()
    second = @opencl launch=false debug_level=2 second_failure()
    for separate_queues in (false, true)
        q1 = cl.CmdQueue()
        q2 = separate_queues ? cl.CmdQueue() : q1
        cl.queue!(q1) do
            first()
        end
        # Order the device work without consuming its report.
        cl.finish(q1; check_exceptions=false)
        cl.queue!(q2) do
            second()
        end
        exc = try
            cl.finish(q2)
            nothing
        catch err
            err
        end
        @test exc isa OpenCL.KernelException
        @test exc.name == "FirstError"
        @test exc.reason == "first kernel"
        @test any(frame -> occursin("first_failure", frame[1]), exc.backtrace)
        @test !any(frame -> occursin("second_failure", frame[1]), exc.backtrace)
        cl.finish(q1)
    end

    function many_failures()
        if isodd(get_local_id())
            OpenCL.@gputhrow "OddError" "odd work-item"
        else
            OpenCL.@gputhrow "EvenError" "even work-item"
        end
        return
    end
    kernel = @opencl launch=false debug_level=2 many_failures()
    for _ in 1:10
        kernel(; global_size=256, local_size=32)
        exc = try
            cl.finish(cl.queue())
            nothing
        catch err
            err
        end
        @test exc isa OpenCL.KernelException
        odd = isodd(exc.work_item[1])
        @test exc.name == (odd ? "OddError" : "EvenError")
        @test exc.reason == (odd ? "odd work-item" : "even work-item")
        @test 1 <= exc.work_group[1] <= 8
        @test !isempty(exc.backtrace)
    end
end

@testset "concurrent exception checks" begin
    fail() = error("concurrent failure")
    kernel = @opencl launch=false debug_level=2 fail()
    dev = cl.device()
    queues = [cl.CmdQueue(), cl.CmdQueue()]
    @sync for queue in queues
        Threads.@spawn begin
            cl.device!(dev)
            cl.queue!(queue) do
                kernel()
            end
        end
    end
    errors = Channel{Any}(length(queues))
    @sync for queue in queues
        Threads.@spawn begin
            err = try
                cl.finish(queue)
                nothing
            catch err
                err
            end
            put!(errors, err)
        end
    end
    reports = [take!(errors) for _ in queues]
    @test count(err -> err isa OpenCL.KernelException, reports) == 1
    @test count(isnothing, reports) == 1
    foreach(cl.finish, queues)
end

@testset "exception mailbox memory" begin
    fail() = error("mailbox backend")
    kernel = @opencl launch=false debug_level=2 fail()
    ctx, dev = cl.context(), cl.device()
    cl.finish(cl.queue())
    key = (ctx, dev)
    old = Base.@lock OpenCL.exception_mailboxes_lock get(OpenCL.exception_mailboxes, key, nothing)
    try
        for backend in OpenCL.exception_mailbox_backends
            supported = backend === :usm ? cl.usm_supported(dev) && cl.usm_capabilities(dev).host.access :
                        backend === :svm_fine ? cl.svm_capabilities(dev).fine_grain_buffer :
                        backend === :svm_coarse ? cl.svm_capabilities(dev).coarse_grain_buffer :
                        cl.bda_supported(dev)
            supported || continue
            @testset "$backend" begin
                mailbox = OpenCL.allocate_exception_mailbox(ctx; backends=(backend,))
                Base.@lock OpenCL.exception_mailboxes_lock OpenCL.exception_mailboxes[key] = mailbox
                for _ in 1:2
                    kernel()
                    exc = try
                        cl.finish(cl.queue())
                        nothing
                    catch err
                        err
                    end
                    @test exc isa OpenCL.KernelException
                    @test exc.dev == dev
                    @test !isempty(exc.backtrace)
                    cl.finish(cl.queue())
                end
            end
        end
        # Exercise the zero-address path without requiring an unsupported physical device.
        mailbox = @test_logs (:warn, r"Device-side exceptions cannot be reported") OpenCL.allocate_exception_mailbox(ctx; backends=())
        Base.@lock OpenCL.exception_mailboxes_lock OpenCL.exception_mailboxes[key] = mailbox
        kernel()
        @test cl.finish(cl.queue()) isa cl.CmdQueue
    finally
        Base.@lock OpenCL.exception_mailboxes_lock begin
            if old === nothing
                delete!(OpenCL.exception_mailboxes, key)
            else
                OpenCL.exception_mailboxes[key] = old
            end
        end
    end
end

@testset "finalizer synchronization" begin
    fail() = error("pending finalizer report")
    @opencl fail()
    queue = cl.queue()
    mailbox = OpenCL.exception_mailboxes[(cl.context(), cl.device())]
    ready, release = Channel{Nothing}(1), Channel{Nothing}(1)
    holder = @async Base.@lock mailbox.lock begin
        put!(ready, nothing)
        take!(release)
    end
    take!(ready)
    try
        # The finalizer path must not wait on the mailbox lock held by another task.
        # Release the lock even on failure so a regression cannot hang the test suite.
        check = @async cl.finish(queue; check_exceptions=false)
        @test timedwait(() -> istaskdone(check), 5) == :ok
    finally
        put!(release, nothing)
        wait(holder)
    end
    @test_throws OpenCL.KernelException cl.finish(queue)
    cl.finish(queue)
end
