module cl

export CLObject

abstract type CLObject end

Base.hash(x::CLObject) = hash(pointer(x))
Base.isequal(x::T, y::T) where {T <: CLObject} = Base.hash(x) == Base.hash(y)
Base.:(==)(x::T, y::T) where {T <: CLObject} = Base.hash(x) == Base.hash(y)

include("api.jl")

# API wrappers
include("error.jl")
include("platform.jl")
include("device.jl")
include("context.jl")
include("cmdqueue.jl")
include("event.jl")
include("memory.jl")
include("buffer.jl")
include("program.jl")
include("kernel.jl")

include("state.jl")

end
