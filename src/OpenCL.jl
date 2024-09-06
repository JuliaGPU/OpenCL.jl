module OpenCL

# library wrappers
include("../lib/CL.jl")
using .cl
export cl

# high-level functionality
include("util.jl")
include("array.jl")

end
