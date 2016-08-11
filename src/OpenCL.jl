module OpenCL

export cl
module cl

abstract CLObject

Base.hash(x::CLObject) = hash(pointer(x))
Base.isequal{T <: CLObject}(x :: T, y :: T) = Base.hash(x) == Base.hash(y)
Base.:(==){T <: CLObject}(x :: T, y :: T) = Base.hash(x) == Base.hash(y)

# OpenCL Types
include("types.jl")

# The arrays contain a nullbyte that we pop first
function CLString(v :: Array{CL_char})
    pop!(v)
    String(reinterpret(UInt8, v))
end

# OpenCL Constants
include("constants.jl")

# OpenCL low level api
include("api.jl")

# Errors
include("error.jl")

# Macros
include("macros.jl")

# OpenCL Platform
include("platform.jl")

# OpenCL Device
include("device.jl")

# OpenCL Context
include("context.jl")

# OpenCL Queue
include("queue.jl")

# OpenCL Event
include("event.jl")

# OpenCL MemObject
include("memory.jl")

# OpenCL Buffer
include("buffer.jl")

# OpenCL Program
include("program.jl")

# OpenCL Kernel
include("kernel.jl")

# Util functions
include("util.jl")

# Multidimensional array
include("array.jl")

@deprecate release! finalize
end # cl
end # module
