module cl

export CLObject, CLString

abstract type CLObject end

Base.hash(x::CLObject) = hash(pointer(x))
Base.isequal(x::T, y::T) where {T <: CLObject} = Base.hash(x) == Base.hash(y)
Base.:(==)(x::T, y::T) where {T <: CLObject} = Base.hash(x) == Base.hash(y)

# The arrays contain a nullbyte that we pop first
function CLString(v::Array{Cchar})
    pop!(v)
    String(reinterpret(UInt8, v))
end

include("api.jl")

# API wrappers
include("error.jl")
include("platform.jl")
include("device.jl")
include("context.jl")
include("queue.jl")
include("event.jl")
include("memory.jl")
include("buffer.jl")
include("program.jl")
include("kernel.jl")

include("state.jl")

end
