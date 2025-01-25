module cl

using Printf

include("pointer.jl")
include("api.jl")

# OpenCL wrapper objects are expected to have an `id` field containing a handle pointer
abstract type CLObject end
Base.pointer(x::CLObject) = x.id
Base.:(==)(a::CLObject, b::CLObject) = pointer(a) == pointer(b)
Base.hash(obj::CLObject, h::UInt) = hash(pointer(obj), h)

# API wrappers
include("error.jl")
include("platform.jl")
include("device.jl")
include("context.jl")
include("cmdqueue.jl")
include("event.jl")
include("memory/memory.jl")
include("buffer.jl")
include("program.jl")
include("kernel.jl")

include("state.jl")

end
