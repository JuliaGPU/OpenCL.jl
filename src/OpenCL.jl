module OpenCL

using Reexport

# library wrappers
include("../lib/CL.jl")
@reexport using .cl
export cl

# high-level functionality
include("util.jl")
include("array.jl")

end
