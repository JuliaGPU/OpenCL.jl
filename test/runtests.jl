using Test
using OpenCL
using pocl_jll

@info "System information:\n" * sprint(io->OpenCL.versioninfo(io))

@testset "OpenCL.jl" begin

@testset "$(platform.name): $(device.name)" for platform in cl.platforms(),
                                                 device in cl.devices(platform)

cl.platform!(platform)
cl.device!(device)

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

end
