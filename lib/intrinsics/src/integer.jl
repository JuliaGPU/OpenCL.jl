# Integer Functions

# TODO: vector types
const generic_integer_types = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64]


# generically typed

for gentype in generic_integer_types
@eval begin

@device_override Base.abs(x::$gentype) = @builtin_ccall("abs", $gentype, ($gentype,), x)
@device_function abs_diff(x::$gentype, y::$gentype) = @builtin_ccall("abs_diff", $gentype, ($gentype, $gentype), x, y)

@device_function add_sat(x::$gentype, y::$gentype) = @builtin_ccall("add_sat", $gentype, ($gentype, $gentype), x, y)
@device_function hadd(x::$gentype, y::$gentype) = @builtin_ccall("hadd", $gentype, ($gentype, $gentype), x, y)
@device_function rhadd(x::$gentype, y::$gentype) = @builtin_ccall("rhadd", $gentype, ($gentype, $gentype), x, y)

@device_override Base.clamp(x::$gentype, minval::$gentype, maxval::$gentype) = @builtin_ccall("clamp", $gentype, ($gentype, $gentype, $gentype), x, minval, maxval)

@device_function clz(x::$gentype) = @builtin_ccall("clz", $gentype, ($gentype,), x)
@device_function ctz(x::$gentype) = @builtin_ccall("ctz", $gentype, ($gentype,), x)

@device_function mad_hi(a::$gentype, b::$gentype, c::$gentype) = @builtin_ccall("mad_hi", $gentype, ($gentype, $gentype, $gentype), a, b, c)
@device_function mad_sat(a::$gentype, b::$gentype, c::$gentype) = @builtin_ccall("mad_sat", $gentype, ($gentype, $gentype, $gentype), a, b, c)

# XXX: these definitions introduce ambiguities
#@device_override Base.max(x::$gentype, y::$gentype) = @builtin_ccall("max", $gentype, ($gentype, $gentype), x, y)
#@device_override Base.min(x::$gentype, y::$gentype) = @builtin_ccall("min", $gentype, ($gentype, $gentype), x, y)

@device_function mul_hi(x::$gentype, y::$gentype) = @builtin_ccall("mul_hi", $gentype, ($gentype, $gentype), x, y)

@device_function rotate(v::$gentype, i::$gentype) = @builtin_ccall("rotate", $gentype, ($gentype, $gentype), v, i)

@device_function sub_sat(x::$gentype, y::$gentype) = @builtin_ccall("sub_sat", $gentype, ($gentype, $gentype), x, y)

@device_function popcount(x::$gentype) = @builtin_ccall("popcount", $gentype, ($gentype,), x)

@device_function mad24(x::$gentype, y::$gentype, z::$gentype) = @builtin_ccall("mad24", $gentype, ($gentype, $gentype, $gentype), x, y, z)
@device_function mul24(x::$gentype, y::$gentype) = @builtin_ccall("mul24", $gentype, ($gentype, $gentype), x, y)

end
end


# specifically typed

@device_function upsample(hi::Cchar, lo::Cuchar) = @builtin_ccall("upsample", Cshort, (Cchar, Cuchar), hi, lo)
upsample(hi::Cuchar, lo::Cuchar) = @builtin_ccall("upsample", Cushort, (Cuchar, Cuchar), hi, lo)
upsample(hi::Cshort, lo::Cushort) = @builtin_ccall("upsample", Cint, (Cshort, Cushort), hi, lo)
upsample(hi::Cushort, lo::Cushort) = @builtin_ccall("upsample", Cuint, (Cushort, Cushort), hi, lo)
upsample(hi::Cint, lo::Cuint) = @builtin_ccall("upsample", Clong, (Cint, Cuint), hi, lo)
upsample(hi::Cuint, lo::Cuint) = @builtin_ccall("upsample", Culong, (Cuint, Cuint), hi, lo)
