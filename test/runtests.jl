module TestOpenCL
using Base.Test

using OpenCL

@testset "aligned convert" begin
    x = ((10f0, 1f0, 2f0), (10f0, 1f0, 2f0), (10f0, 1f0, 2f0))
    x_aligned = cl.packed_convert(x)

    @test x_aligned == ((10f0, 1f0, 2f0), cl.Pad{4}(), (10f0, 1f0, 2f0), cl.Pad{4}(), (10f0, 1f0, 2f0), cl.Pad{4}())
    x_aligned_t = cl.packed_convert(typeof(x))
    @test x_aligned_t == typeof(x_aligned)

    x = cl.packed_convert(77f0)
    @test x == 77f0
end

function create_test_buffer()
    ctx = cl.create_some_context()
    queue = cl.CmdQueue(ctx)
    testarray = zeros(Float32, 1000)
    buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
    return (queue, buf, testarray, ctx)
end

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
include("test_array.jl")

@testset "context jl reference counting" begin
    gc()
    @test isempty(cl._ctx_reference_count)
end

end # module
