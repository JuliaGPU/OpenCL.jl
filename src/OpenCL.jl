module OpenCL

# library wrappers
include("../lib/CL.jl")
using .cl
export cl

# high-level functionality
include("util.jl")
include("array.jl")

function __init__()
    if cl.libopencl == ""
        @warn "Could not locate an OpenCL library\nOpenCL API calls will be unavailable"
    end
end

end
