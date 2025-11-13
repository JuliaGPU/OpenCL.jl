module SPIRVIntrinsicsSIMDExt

using SPIRVIntrinsics
using SPIRVIntrinsics: @device_override, @device_function, @builtin_ccall, @typed_ccall
using SIMD
import SpecialFunctions

const known_intrinsics = String[]

# Generate vectorized math intrinsics
for N in [2, 3, 4, 8, 16], T in [Float16, Float32, Float64]
    VT = :(Vec{$N,$T})
    LVT = :(SIMD.LVec{$N,$T})

    @eval begin
        # Unary operations
        @device_override @inline Base.acos(x::$VT) = $VT(@builtin_ccall("acos", $LVT, ($LVT,), x.data))
        @device_override @inline Base.acosh(x::$VT) = $VT(@builtin_ccall("acosh", $LVT, ($LVT,), x.data))
        @device_function @inline SPIRVIntrinsics.acospi(x::$VT) = $VT(@builtin_ccall("acospi", $LVT, ($LVT,), x.data))

        @device_override @inline Base.asin(x::$VT) = $VT(@builtin_ccall("asin", $LVT, ($LVT,), x.data))
        @device_override @inline Base.asinh(x::$VT) = $VT(@builtin_ccall("asinh", $LVT, ($LVT,), x.data))
        @device_function @inline SPIRVIntrinsics.asinpi(x::$VT) = $VT(@builtin_ccall("asinpi", $LVT, ($LVT,), x.data))

        @device_override @inline Base.atan(x::$VT) = $VT(@builtin_ccall("atan", $LVT, ($LVT,), x.data))
        @device_override @inline Base.atanh(x::$VT) = $VT(@builtin_ccall("atanh", $LVT, ($LVT,), x.data))
        @device_function @inline SPIRVIntrinsics.atanpi(x::$VT) = $VT(@builtin_ccall("atanpi", $LVT, ($LVT,), x.data))

        @device_override @inline Base.cbrt(x::$VT) = $VT(@builtin_ccall("cbrt", $LVT, ($LVT,), x.data))
        @device_override @inline Base.ceil(x::$VT) = $VT(@builtin_ccall("ceil", $LVT, ($LVT,), x.data))

        @device_override @inline Base.cos(x::$VT) = $VT(@builtin_ccall("cos", $LVT, ($LVT,), x.data))
        @device_override @inline Base.cosh(x::$VT) = $VT(@builtin_ccall("cosh", $LVT, ($LVT,), x.data))
        @device_override @inline Base.cospi(x::$VT) = $VT(@builtin_ccall("cospi", $LVT, ($LVT,), x.data))

        @device_override @inline SpecialFunctions.erfc(x::$VT) = $VT(@builtin_ccall("erfc", $LVT, ($LVT,), x.data))
        @device_override @inline SpecialFunctions.erf(x::$VT) = $VT(@builtin_ccall("erf", $LVT, ($LVT,), x.data))

        @device_override @inline Base.exp(x::$VT) = $VT(@builtin_ccall("exp", $LVT, ($LVT,), x.data))
        @device_override @inline Base.exp2(x::$VT) = $VT(@builtin_ccall("exp2", $LVT, ($LVT,), x.data))
        @device_override @inline Base.exp10(x::$VT) = $VT(@builtin_ccall("exp10", $LVT, ($LVT,), x.data))
        @device_override @inline Base.expm1(x::$VT) = $VT(@builtin_ccall("expm1", $LVT, ($LVT,), x.data))

        @device_override @inline Base.abs(x::$VT) = $VT(@builtin_ccall("fabs", $LVT, ($LVT,), x.data))
        @device_override @inline Base.floor(x::$VT) = $VT(@builtin_ccall("floor", $LVT, ($LVT,), x.data))

        @device_override @inline SpecialFunctions.loggamma(x::$VT) = $VT(@builtin_ccall("lgamma", $LVT, ($LVT,), x.data))

        @device_override @inline Base.log(x::$VT) = $VT(@builtin_ccall("log", $LVT, ($LVT,), x.data))
        @device_override @inline Base.log2(x::$VT) = $VT(@builtin_ccall("log2", $LVT, ($LVT,), x.data))
        @device_override @inline Base.log10(x::$VT) = $VT(@builtin_ccall("log10", $LVT, ($LVT,), x.data))
        @device_override @inline Base.log1p(x::$VT) = $VT(@builtin_ccall("log1p", $LVT, ($LVT,), x.data))
        @device_function @inline SPIRVIntrinsics.logb(x::$VT) = $VT(@builtin_ccall("logb", $LVT, ($LVT,), x.data))

        @device_function @inline SPIRVIntrinsics.rint(x::$VT) = $VT(@builtin_ccall("rint", $LVT, ($LVT,), x.data))
        @device_override @inline Base.round(x::$VT) = $VT(@builtin_ccall("round", $LVT, ($LVT,), x.data))
        @device_function @inline SPIRVIntrinsics.rsqrt(x::$VT) = $VT(@builtin_ccall("rsqrt", $LVT, ($LVT,), x.data))

        @device_override @inline Base.sin(x::$VT) = $VT(@builtin_ccall("sin", $LVT, ($LVT,), x.data))
        @device_override @inline Base.sinh(x::$VT) = $VT(@builtin_ccall("sinh", $LVT, ($LVT,), x.data))
        @device_override @inline Base.sinpi(x::$VT) = $VT(@builtin_ccall("sinpi", $LVT, ($LVT,), x.data))

        @device_override @inline Base.sqrt(x::$VT) = $VT(@builtin_ccall("sqrt", $LVT, ($LVT,), x.data))

        @device_override @inline Base.tan(x::$VT) = $VT(@builtin_ccall("tan", $LVT, ($LVT,), x.data))
        @device_override @inline Base.tanh(x::$VT) = $VT(@builtin_ccall("tanh", $LVT, ($LVT,), x.data))
        @device_override @inline Base.tanpi(x::$VT) = $VT(@builtin_ccall("tanpi", $LVT, ($LVT,), x.data))

        @device_override @inline SpecialFunctions.gamma(x::$VT) = $VT(@builtin_ccall("tgamma", $LVT, ($LVT,), x.data))

        @device_override @inline Base.trunc(x::$VT) = $VT(@builtin_ccall("trunc", $LVT, ($LVT,), x.data))

        # Binary operations
        @device_override @inline Base.atan(y::$VT, x::$VT) = $VT(@builtin_ccall("atan2", $LVT, ($LVT, $LVT), y.data, x.data))
        @device_function @inline SPIRVIntrinsics.atanpi(y::$VT, x::$VT) = $VT(@builtin_ccall("atan2pi", $LVT, ($LVT, $LVT), y.data, x.data))

        @device_override @inline Base.copysign(x::$VT, y::$VT) = $VT(@builtin_ccall("copysign", $LVT, ($LVT, $LVT), x.data, y.data))
        @device_function @inline SPIRVIntrinsics.dim(x::$VT, y::$VT) = $VT(@builtin_ccall("fdim", $LVT, ($LVT, $LVT), x.data, y.data))

        @device_override @inline Base.hypot(x::$VT, y::$VT) = $VT(@builtin_ccall("hypot", $LVT, ($LVT, $LVT), x.data, y.data))

        @device_override @inline Base.max(x::$VT, y::$VT) = $VT(@builtin_ccall("fmax", $LVT, ($LVT, $LVT), x.data, y.data))
        @device_override @inline Base.min(x::$VT, y::$VT) = $VT(@builtin_ccall("fmin", $LVT, ($LVT, $LVT), x.data, y.data))

        @device_function @inline SPIRVIntrinsics.maxmag(x::$VT, y::$VT) = $VT(@builtin_ccall("maxmag", $LVT, ($LVT, $LVT), x.data, y.data))
        @device_function @inline SPIRVIntrinsics.minmag(x::$VT, y::$VT) = $VT(@builtin_ccall("minmag", $LVT, ($LVT, $LVT), x.data, y.data))

        @device_function @inline SPIRVIntrinsics.nextafter(x::$VT, y::$VT) = $VT(@builtin_ccall("nextafter", $LVT, ($LVT, $LVT), x.data, y.data))

        @device_override @inline Base.:(^)(x::$VT, y::$VT) = $VT(@builtin_ccall("pow", $LVT, ($LVT, $LVT), x.data, y.data))
        @device_function @inline SPIRVIntrinsics.powr(x::$VT, y::$VT) = $VT(@builtin_ccall("powr", $LVT, ($LVT, $LVT), x.data, y.data))

        @device_override @inline Base.rem(x::$VT, y::$VT) = $VT(@builtin_ccall("remainder", $LVT, ($LVT, $LVT), x.data, y.data))

        # Ternary operations
        @device_override @inline Base.fma(a::$VT, b::$VT, c::$VT) = $VT(@builtin_ccall("fma", $LVT, ($LVT, $LVT, $LVT), a.data, b.data, c.data))
        @device_function @inline SPIRVIntrinsics.mad(a::$VT, b::$VT, c::$VT) = $VT(@builtin_ccall("mad", $LVT, ($LVT, $LVT, $LVT), a.data, b.data, c.data))
    end

    # Special operations with Int32 parameters
    VIntT = :(Vec{$N,Int32})
    LVIntT = :(SIMD.LVec{$N,Int32})

    @eval begin
        @device_function @inline SPIRVIntrinsics.ilogb(x::$VT) = $VIntT(@builtin_ccall("ilogb", $LVIntT, ($LVT,), x.data))
        @device_override @inline Base.ldexp(x::$VT, k::$VIntT) = $VT(@builtin_ccall("ldexp", $LVT, ($LVT, $LVIntT), x.data, k.data))
        @device_override @inline Base.:(^)(x::$VT, y::$VIntT) = $VT(@builtin_ccall("pown", $LVT, ($LVT, $LVIntT), x.data, y.data))
        @device_function @inline SPIRVIntrinsics.rootn(x::$VT, y::$VIntT) = $VT(@builtin_ccall("rootn", $LVT, ($LVT, $LVIntT), x.data, y.data))
    end
end

# nan functions - take unsigned integer codes and return floats
for N in [2, 3, 4, 8, 16]
    @eval begin
        @device_function @inline SPIRVIntrinsics.nan(nancode::Vec{$N,UInt16}) = Vec{$N,Float16}(@builtin_ccall("nan", SIMD.LVec{$N,Float16}, (SIMD.LVec{$N,UInt16},), nancode.data))
        @device_function @inline SPIRVIntrinsics.nan(nancode::Vec{$N,UInt32}) = Vec{$N,Float32}(@builtin_ccall("nan", SIMD.LVec{$N,Float32}, (SIMD.LVec{$N,UInt32},), nancode.data))
        @device_function @inline SPIRVIntrinsics.nan(nancode::Vec{$N,UInt64}) = Vec{$N,Float64}(@builtin_ccall("nan", SIMD.LVec{$N,Float64}, (SIMD.LVec{$N,UInt64},), nancode.data))
    end
end

end # module
