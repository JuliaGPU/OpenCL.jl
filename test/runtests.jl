module TestOpenCL
using Base.Test

using OpenCL


test_source = "
struct __attribute__((packed)) Test2{
    long f1;
    int __attribute__((aligned (8))) f2;
};

__kernel void structest(__global float *out, struct Test2 b){
    out[0] = b.f1;
    out[1] = b.f2;
}
"
for device in cl.devices()
    if device[:platform][:name] == "Portable Computing Language"
        warn("Skipping OpenCL.Kernel constructor for " *
             "Portable Computing Language Platform")
        continue
    end
    ctx = cl.Context(device)
    prg = cl.Program(ctx, source = test_source)
    queue = cl.CmdQueue(ctx)
    cl.build!(prg)
    structkernel = cl.Kernel(prg, "structest")
    out = cl.Buffer(Float32, ctx, :w, 6)
    bstruct = (1, Int32(4))
    structkernel[queue, (1,)](out, bstruct)
    r = cl.read(queue, out)
    println(r[1:2])
end



@testset "aligned convert" begin
    x = ((10f0, 1f0, 2f0), (10f0, 1f0, 2f0), (10f0, 1f0, 2f0))
    clx = cl.replace_different_layout(x)

    @test clx == ((10f0, 1f0, 2f0, 0f0), (10f0, 1f0, 2f0, 0f0), (10f0, 1f0, 2f0, 0f0))
    x = (nothing, nothing, nothing)
    clx = cl.replace_different_layout(x)
    @test clx == (0,0,0)
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
