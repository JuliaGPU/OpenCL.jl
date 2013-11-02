module OpenCL

# OpenCL Types 
include("types.jl")

# OpenCL Constants 
include("constants.jl")

# OpenCL low level api 
include("api.jl")

# util functions 
include("util.jl")

# --- Errors ---
include("error.jl")

# --- Macros ---
include("macros.jl")

# --- OpenCL Platform --- 
include("platform.jl")

# --- OpenCL Device --- 
include("device.jl")

# --- OpenCL Context ---
include("context.jl")

# --- OpenCL Queue ---
include("queue.jl")

end # module
