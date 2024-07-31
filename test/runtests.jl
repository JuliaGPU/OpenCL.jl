module TestOpenCL
using Test
using OpenCL
using Base.GC

# Use POCL for the tests
# XXX: support testing with other OpenCL implementations
using pocl_jll
platform = filter(cl.platforms()) do platform
    cl.info(platform, :name) == "Portable Computing Language"
end |> first
device = first(cl.devices(platform, :cpu))
@info "Testing using POCL" platform device

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
include("minver.jl")
#include("event.jl")
include("program.jl")
include("kernel.jl")
include("behaviour.jl")
include("memory.jl")
include("buffer.jl")
include("array.jl")

@testset "context jl reference counting" begin
    Base.GC.gc()
    @test isempty(cl._ctx_reference_count)
end

end

end # module
