module TestOpenCL
    using FactCheck
    using Base.Test
    using Compat

    import OpenCL
    const cl = OpenCL

    if VERSION < v"0.4.0-dev+1969"
        finalize(x) = nothing
    end

    FactCheck.onlystats(true)

    function create_test_buffer()
        ctx = cl.create_some_context()
        queue = cl.CmdQueue(ctx)
        testarray = zeros(Float32, 1000)
        buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
        return (queue, buf, testarray, ctx)
    end

    # include("test_platform.jl")
    # include("test_context.jl")
    # include("test_device.jl")
    # include("test_cmdqueue.jl")
    # include("test_macros.jl")
    # include("test_event.jl")
    # include("test_program.jl")
    # include("test_kernel.jl")
    # include("test_behaviour.jl")
    include("test_array.jl")
    include("test_memory.jl")
    include("test_buffer.jl")

    FactCheck.exitstatus()

end # module
