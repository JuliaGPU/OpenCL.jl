module OpenCL

export cl
module cl

abstract type CLObject end

Base.hash(x::CLObject) = hash(pointer(x))
Base.isequal(x :: T, y :: T) where {T <: CLObject} = Base.hash(x) == Base.hash(y)
Base.:(==)(x :: T, y :: T) where {T <: CLObject} = Base.hash(x) == Base.hash(y)

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

function __init__()
    if cl.api.libopencl == ""
        @warn "Could not locate an OpenCL library, this package will not work!"
    end
end

end # module
