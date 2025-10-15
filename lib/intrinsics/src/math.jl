# Math Functions

# TODO: vector types
const generic_types = [Float16, Float32, Float64]
const generic_types_float = [Float32]
const generic_types_double = [Float64]


# generically typed

for gentype in generic_types
@eval begin

@device_override Base.acos(x::$gentype) = @builtin_ccall("acos", $gentype, ($gentype,), x)
@device_override Base.acosh(x::$gentype) = @builtin_ccall("acosh", $gentype, ($gentype,), x)
@device_function acospi(x::$gentype) = @builtin_ccall("acospi", $gentype, ($gentype,), x)

@device_override Base.asin(x::$gentype) = @builtin_ccall("asin", $gentype, ($gentype,), x)
@device_override Base.asinh(x::$gentype) = @builtin_ccall("asinh", $gentype, ($gentype,), x)
@device_function asinpi(x::$gentype) = @builtin_ccall("asinpi", $gentype, ($gentype,), x)

@device_override Base.atan(y_over_x::$gentype) = @builtin_ccall("atan", $gentype, ($gentype,), y_over_x)
@device_override Base.atan(y::$gentype, x::$gentype) = @builtin_ccall("atan2", $gentype, ($gentype, $gentype), y, x)
@device_override Base.atanh(x::$gentype) = @builtin_ccall("atanh", $gentype, ($gentype,), x)
@device_function atanpi(x::$gentype) = @builtin_ccall("atanpi", $gentype, ($gentype,), x)
@device_function atanpi(y::$gentype, x::$gentype) = @builtin_ccall("atan2pi", $gentype, ($gentype, $gentype), y, x)

@device_override Base.cbrt(x::$gentype) = @builtin_ccall("cbrt", $gentype, ($gentype,), x)

@device_override Base.ceil(x::$gentype) = @builtin_ccall("ceil", $gentype, ($gentype,), x)

@device_override Base.copysign(x::$gentype, y::$gentype) = @builtin_ccall("copysign", $gentype, ($gentype, $gentype), x, y)

@device_override Base.cos(x::$gentype) = @builtin_ccall("cos", $gentype, ($gentype,), x)
@device_override Base.cosh(x::$gentype) = @builtin_ccall("cosh", $gentype, ($gentype,), x)
@device_override Base.cospi(x::$gentype) = @builtin_ccall("cospi", $gentype, ($gentype,), x)

@device_override SpecialFunctions.erfc(x::$gentype) = @builtin_ccall("erfc", $gentype, ($gentype,), x)
@device_override SpecialFunctions.erf(x::$gentype) = @builtin_ccall("erf", $gentype, ($gentype,), x)

@device_override Base.exp(x::$gentype) = @builtin_ccall("exp", $gentype, ($gentype,), x)
@device_override Base.exp2(x::$gentype) = @builtin_ccall("exp2", $gentype, ($gentype,), x)
@device_override Base.exp10(x::$gentype) = @builtin_ccall("exp10", $gentype, ($gentype,), x)
@device_override Base.expm1(x::$gentype) = @builtin_ccall("expm1", $gentype, ($gentype,), x)

@device_override Base.abs(x::$gentype) = @builtin_ccall("fabs", $gentype, ($gentype,), x)

@device_function dim(x::$gentype, y::$gentype) = @builtin_ccall("fdim", $gentype, ($gentype, $gentype), x, y)

@device_override Base.floor(x::$gentype) = @builtin_ccall("floor", $gentype, ($gentype,), x)

@device_override Base.fma(a::$gentype, b::$gentype, c::$gentype) = @builtin_ccall("fma", $gentype, ($gentype, $gentype, $gentype), a, b, c)

@device_override Base.max(x::$gentype, y::$gentype) = @builtin_ccall("fmax", $gentype, ($gentype, $gentype), x, y)

@device_override Base.min(x::$gentype, y::$gentype) = @builtin_ccall("fmin", $gentype, ($gentype, $gentype), x, y)

# NOTE: Julia's mod behaves differently than fmod
#@device_override Base.mod(x::$gentype, y::$gentype) = @builtin_ccall("fmod", $gentype, ($gentype, $gentype), x, y)
# fract(x::$gentype, $gentype *iptr) = @builtin_ccall("fract", $gentype, ($gentype, $gentype *), x, iptr)

# TODO: remove once https://github.com/pocl/pocl/issues/2034 is addressed
if $gentype != Float16
    @device_override Base.hypot(x::$gentype, y::$gentype) = @builtin_ccall("hypot", $gentype, ($gentype, $gentype), x, y)
end

@device_override SpecialFunctions.loggamma(x::$gentype) = @builtin_ccall("lgamma", $gentype, ($gentype,), x)

@device_override Base.log(x::$gentype) = @builtin_ccall("log", $gentype, ($gentype,), x)
@device_override Base.log2(x::$gentype) = @builtin_ccall("log2", $gentype, ($gentype,), x)
@device_override Base.log10(x::$gentype) = @builtin_ccall("log10", $gentype, ($gentype,), x)
@device_override Base.log1p(x::$gentype) = @builtin_ccall("log1p", $gentype, ($gentype,), x)
@device_function logb(x::$gentype) = @builtin_ccall("logb", $gentype, ($gentype,), x)

@device_function mad(a::$gentype, b::$gentype, c::$gentype) = @builtin_ccall("mad", $gentype, ($gentype, $gentype, $gentype), a, b, c)

@device_function maxmag(x::$gentype, y::$gentype) = @builtin_ccall("maxmag", $gentype, ($gentype, $gentype), x, y)
@device_function minmag(x::$gentype, y::$gentype) = @builtin_ccall("minmag", $gentype, ($gentype, $gentype), x, y)

# modf(x::$gentype, $gentype *iptr) = @builtin_ccall("modf", $gentype, ($gentype, $gentype *), x, iptr)

@device_function nextafter(x::$gentype, y::$gentype) = @builtin_ccall("nextafter", $gentype, ($gentype, $gentype), x, y)

@device_override Base.:(^)(x::$gentype, y::$gentype) = @builtin_ccall("pow", $gentype, ($gentype, $gentype), x, y)
@device_function powr(x::$gentype, y::$gentype) = @builtin_ccall("powr", $gentype, ($gentype, $gentype), x, y)

@device_function rint(x::$gentype) = @builtin_ccall("rint", $gentype, ($gentype,), x)

@device_override Base.round(x::$gentype) = @builtin_ccall("round", $gentype, ($gentype,), x)

@device_function rsqrt(x::$gentype) = @builtin_ccall("rsqrt", $gentype, ($gentype,), x)

@device_override Base.sin(x::$gentype) = @builtin_ccall("sin", $gentype, ($gentype,), x)
@device_override function Base.sincos(x::$gentype)
    cosval = Ref{$gentype}()
    sinval = GC.@preserve cosval begin
        ptr = Base.unsafe_convert(Ptr{$gentype}, cosval)
        llvm_ptr = reinterpret(LLVMPtr{$gentype, AS.Function}, ptr)
        @builtin_ccall("sincos", $gentype, ($gentype, LLVMPtr{$gentype, AS.Function}), x, llvm_ptr)
    end
    return sinval, cosval[]
end
@device_override Base.sinh(x::$gentype) = @builtin_ccall("sinh", $gentype, ($gentype,), x)
@device_override Base.sinpi(x::$gentype) = @builtin_ccall("sinpi", $gentype, ($gentype,), x)

@device_override Base.sqrt(x::$gentype) = @builtin_ccall("sqrt", $gentype, ($gentype,), x)

@device_override Base.tan(x::$gentype) = @builtin_ccall("tan", $gentype, ($gentype,), x)
@device_override Base.tanh(x::$gentype) = @builtin_ccall("tanh", $gentype, ($gentype,), x)
@device_override Base.tanpi(x::$gentype) = @builtin_ccall("tanpi", $gentype, ($gentype,), x)

@device_override SpecialFunctions.gamma(x::$gentype) = @builtin_ccall("tgamma", $gentype, ($gentype,), x)

@device_override Base.trunc(x::$gentype) = @builtin_ccall("trunc", $gentype, ($gentype,), x)

end
end


# generically typed -- only floats

for gentypef in generic_types_float

if gentypef !== Float32
@eval begin
@device_override Base.max(x::$gentypef, y::Float32) = @builtin_ccall("fmax", $gentypef, ($gentypef, Float32), x, y)
@device_override Base.min(x::$gentypef, y::Float32) = @builtin_ccall("fmin", $gentypef, ($gentypef, Float32), x, y)
end
end

end


# generically typed -- only doubles

for gentyped in generic_types_double

if gentyped !== Float64
@eval begin
@device_override Base.min(x::$gentyped, y::Float64) = @builtin_ccall("fmin", $gentyped, ($gentyped, Float64), x, y)
@device_override Base.max(x::$gentyped, y::Float64) = @builtin_ccall("fmax", $gentyped, ($gentyped, Float64), x, y)
end
end

end


# specifically typed

# frexp(x::Float32{n}, Int32{n} *exp) = @builtin_ccall("frexp", Float32{n}, (Float32{n}, Int32{n} *), x, exp)
# frexp(x::Float32, Int32 *exp) = @builtin_ccall("frexp", Float32, (Float32, Int32 *), x, exp)
# frexp(x::Float64{n}, Int32{n} *exp) = @builtin_ccall("frexp", Float64{n}, (Float64{n}, Int32{n} *), x, exp)
# frexp(x::Float64, Int32 *exp) = @builtin_ccall("frexp", Float64, (Float64, Int32 *), x, exp)

@device_function ilogb(x::Float16) = @builtin_ccall("ilogb", Int32, (Float16,), x)
# ilogb(x::Float32{n}) = @builtin_ccall("ilogb", Int32{n}, (Float32{n},), x)
@device_function ilogb(x::Float32) = @builtin_ccall("ilogb", Int32, (Float32,), x)
# ilogb(x::Float64{n}) = @builtin_ccall("ilogb", Int32{n}, (Float64{n},), x)
@device_function ilogb(x::Float64) = @builtin_ccall("ilogb", Int32, (Float64,), x)

@device_override Base.ldexp(x::Float16, k::Int32) = @builtin_ccall("ldexp", Float16, (Float16, Int32), x, k)
# ldexp(x::Float32{n}, k::Int32{n}) = @builtin_ccall("ldexp", Float32{n}, (Float32{n}, Int32{n}), x, k)
# ldexp(x::Float32{n}, k::Int32) = @builtin_ccall("ldexp", Float32{n}, (Float32{n}, Int32), x, k)
@device_override Base.ldexp(x::Float32, k::Int32) = @builtin_ccall("ldexp", Float32, (Float32, Int32), x, k)
# ldexp(x::Float64{n}, k::Int32{n}) = @builtin_ccall("ldexp", Float64{n}, (Float64{n}, Int32{n}), x, k)
# ldexp(x::Float64{n}, k::Int32) = @builtin_ccall("ldexp", Float64{n}, (Float64{n}, Int32), x, k)
@device_override Base.ldexp(x::Float64, k::Int32) = @builtin_ccall("ldexp", Float64, (Float64, Int32), x, k)

# lgamma_r(x::Float32{n}, Int32{n} *signp) = @builtin_ccall("lgamma_r", Float32{n}, (Float32{n}, Int32{n} *), x, signp)
# lgamma_r(x::Float32, Int32 *signp) = @builtin_ccall("lgamma_r", Float32, (Float32, Int32 *), x, signp)
# lgamma_r(x::Float64{n}, Int32{n} *signp) = @builtin_ccall("lgamma_r", Float64{n}, (Float64{n}, Int32{n} *), x, signp)
# Float64 lgamma_r(x::Float64, Int32 *signp) = @builtin_ccall("lgamma_r", Float64, (Float64, Int32 *), x, signp)

@device_function nan(nancode::UInt16) = @builtin_ccall("nan", Float16, (UInt16,), nancode)
# nan(nancode::uintn) = @builtin_ccall("nan", Float32{n}, (uintn,), nancode)
@device_function nan(nancode::UInt32) = @builtin_ccall("nan", Float32, (UInt32,), nancode)
# nan(nancode::UInt64{n}) = @builtin_ccall("nan", Float64{n}, (UInt64{n},), nancode)
@device_function nan(nancode::UInt64) = @builtin_ccall("nan", Float64, (UInt64,), nancode)

@device_override Base.:(^)(x::Float16, y::Int32) = @builtin_ccall("pown", Float16, (Float16, Int32), x, y)
# pown(x::Float32{n}, y::Int32{n}) = @builtin_ccall("pown", Float32{n}, (Float32{n}, Int32{n}), x, y)
@device_override Base.:(^)(x::Float32, y::Int32) = @builtin_ccall("pown", Float32, (Float32, Int32), x, y)
# pown(x::Float64{n}, y::Int32{n}) = @builtin_ccall("pown", Float64{n}, (Float64{n}, Int32{n}), x, y)
@device_override Base.:(^)(x::Float64, y::Int32) = @builtin_ccall("pown", Float64, (Float64, Int32), x, y)

# remquo(x::Float32{n}, y::Float32{n}, Int32{n} *quo) = @builtin_ccall("remquo", Float32{n}, (Float32{n}, Float32{n}, Int32{n} *), x, y, quo)
# remquo(x::Float32, y::Float32, Int32 *quo) = @builtin_ccall("remquo", Float32, (Float32, Float32, Int32 *), x::Float32, y, quo)
# remquo(x::Float64{n}, y::Float64{n}, Int32{n} *quo) = @builtin_ccall("remquo", Float64{n}, (Float64{n}, Float64{n}, Int32{n} *), x, y, quo)
# remquo(x::Float64, y::Float64, Int32 *quo) = @builtin_ccall("remquo", Float64, (Float64, Float64, Int32 *), x, y, quo)

@device_function rootn(x::Float16, y::Int32) = @builtin_ccall("rootn", Float16, (Float16, Int32), x, y)
# rootn(x::Float32{n}, y::Int32{n}) = @builtin_ccall("rootn", Float32{n}, (Float32{n}, Int32{n}), x, y)
@device_function rootn(x::Float32, y::Int32) = @builtin_ccall("rootn", Float32, (Float32, Int32), x, y)
# rootn(x::Float64{n}, y::Int32{n}) = @builtin_ccall("rootn", Float64{n}, (Float64{n}, Int32{n}), x, y)
@device_function rootn(x::Float64, y::Int32) = @builtin_ccall("rootn", Float64, (Float64, Int32), x, y)


# TODO: half and native

function _mulhi(a::Int64, b::Int64)
    shift = sizeof(a) * 4
    mask = typemax(UInt32)
    a1, a2 = (a >> shift), a & mask
    b1, b2 = (b >> shift), b & mask
    a1b1, a1b2, a2b1 = a1*b1, a1*b2, a2*b1
    t1 = a1b2 + _mulhi(a2 % UInt32, b2 % UInt32)
    t2 = a2b1 + (t1 & mask)
    a1b1 + (t1 >> shift) + (t2 >> shift)
end
@static if isdefined(Base.MultiplicativeInverses, :_mul_high)
    _mulhi(a::T, b::T) where {T<:Union{Signed, Unsigned}} = Base.MultiplicativeInverses._mul_high(a, b)
    @device_override Base.MultiplicativeInverses._mul_high(a::Int64, b::Int64) = _mulhi(a, b)
else
    _mulhi(a::T, b::T) where {T<:Union{Signed, Unsigned}} = ((widen(a)*b) >>> (sizeof(a)*8)) % T
    @device_override function Base.div(a::Int64, b::Base.MultiplicativeInverses.SignedMultiplicativeInverse{Int64})
        x = _mulhi(a, b.multiplier)
        x += (a*b.addmul) % Int64
        ifelse(abs(b.divisor) == 1, a*b.divisor, (signbit(x) + (x >> b.shift)) % Int64)
    end
end
