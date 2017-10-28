module TestOpenCL
using Base.Test

using OpenCL

immutable CLTestStruct
    f1::NTuple{3, Float32}
    f2::Void
    f3::Float32
end

test_source = "
//packed
struct __attribute__((packed)) Test{
    float3 f1;
    int f2; // empty type gets replaced with Int32 (no empty types allowed in OpenCL)
    // you might need to define the alignement of fields to match julia's layout
    float f3; // for the types used here the alignement matches though!
};

__kernel void structest(__global float *out, struct Test a){
    out[0] = a.f1.x;
    out[1] = a.f1.y;
    out[2] = a.f1.z;
    out[3] = a.f3;
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
    out = cl.Buffer(Float32, ctx, :w, 4)
    astruct = CLTestStruct((1f0, 2f0, 3f0), nothing, 22f0)
    structkernel[queue, (1,)](out, astruct)
    r = cl.read(queue, out)
    @test r == [1f0, 2f0, 3f0, 22f0]
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
