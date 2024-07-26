#=== TypeAliases ===#

# Scalar types
const CL_half   = Float16
const CL_float  = Float32
const CL_double = Float64


#=== Conversion Functions ===#

cl_int(x)      = Int32(x)
cl_uint(x)     = UInt32(x)
cl_ulong(x)    = UInt64(x)

cl_half(x)     = UInt16(x)
cl_float(x)    = Float32(x)
cl_double(x)   = Float64(x)

cl_bool(x)     = x != 0 ? cl_uint(1) : cl_uint(0)
cl_bitfield(x) = cl_ulong(x)
