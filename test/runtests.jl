using Test
using OpenCL

backend = lowercase(get(ENV, "JULIA_OPENCL_BACKEND", "pocl"))
if backend == "pocl"
    using pocl_jll
end
cl.platform!(backend)
@info """Testing using $backend back-end
         - platform: $(cl.info(cl.platform(), :name))
         - device: $(cl.info(cl.device(), :name))

         To test with a different back-end, define JULIA_OPENCL_BACKEND."""

@testset "OpenCL.jl" begin

@testset "layout" begin
    x = ((10f0, 1f0, 2f0), (10f0, 1f0, 2f0), (10f0, 1f0, 2f0))
    clx = cl.replace_different_layout(x)

    @test clx == ((10f0, 1f0, 2f0, 0f0), (10f0, 1f0, 2f0, 0f0), (10f0, 1f0, 2f0, 0f0))
    x = (nothing, nothing, nothing)
    clx = cl.replace_different_layout(x)
    @test clx == 0 # TODO should it be like this?
end

include("platform.jl")
include("context.jl")
include("device.jl")
include("cmdqueue.jl")
include("event.jl")
include("program.jl")
include("kernel.jl")
include("behaviour.jl")
include("memory.jl")
include("buffer.jl")
include("array.jl")

end
