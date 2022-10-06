module TestOpenCL
using Test
using OpenCL
using Base.GC

@testset "layout" begin
    x = ((10f0, 1f0, 2f0), (10f0, 1f0, 2f0), (10f0, 1f0, 2f0))
    clx = cl.replace_different_layout(x)

    @test clx == ((10f0, 1f0, 2f0, 0f0), (10f0, 1f0, 2f0, 0f0), (10f0, 1f0, 2f0, 0f0))
    x = (nothing, nothing, nothing)
    clx = cl.replace_different_layout(x)
    @test clx == 0 # TODO should it be like this?
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
include("test_minver.jl")
#TODO: fix test_event.jl
#include("test_event.jl")
include("test_program.jl")
include("test_kernel.jl")
include("test_behaviour.jl")
include("test_memory.jl")
include("test_buffer.jl")
include("test_array.jl")

@testset "context jl reference counting" begin
    Base.GC.gc()
    @test isempty(cl._ctx_reference_count)
end

end # module
