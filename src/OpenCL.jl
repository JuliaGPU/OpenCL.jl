module OpenCL

# OpenCL Types 
include("types.jl")

# OpenCL Constants 
include("constants.jl")

# OpenCL low level api 
include("api.jl")

@linux_only begin
    const libopencl = "libOpenCL"
end

# --- Macros ---
include("macros.jl")

# --- OpenCL Platform --- 
include("platform.jl")

# --- OpenCL Device --- 
include("device.jl")

# --- OpenCL Context ---
#include("context.jl")

# --- OpenCL Queue ---
#include("queue.jl")

end # module
