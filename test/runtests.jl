module TestOpenCL
    using FactCheck
    using Base.Test

    import OpenCL
    const cl = OpenCL

    include("test_platform.jl")
    include("test_context.jl")
    include("test_device.jl")
    include("test_cmdqueue.jl")
    include("test_macros.jl")
    include("test_event.jl")
    include("test_program.jl")
    include("test_kernel.jl")
    include("test_behaviour.jl")
    include("test_memory.jl")
    include("test_buffer.jl")

    FactCheck.exitstatus()
end # module