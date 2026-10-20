module SPIRVIntrinsicsSIMDSpecialFunctionsExt

using SPIRVIntrinsics
using SPIRVIntrinsics: @device_override, @builtin_ccall, @typed_ccall, method_table
using SIMD
import SpecialFunctions

const known_intrinsics = String[]

for N in [2, 3, 4, 8, 16], T in [Float16, Float32, Float64]
    VT = :(Vec{$N,$T})
    LVT = :(SIMD.LVec{$N,$T})

    @eval begin
        @device_override @inline SpecialFunctions.erfc(x::$VT) = $VT(@builtin_ccall("erfc", $LVT, ($LVT,), x.data))
        @device_override @inline SpecialFunctions.erf(x::$VT) = $VT(@builtin_ccall("erf", $LVT, ($LVT,), x.data))

        @device_override @inline SpecialFunctions.loggamma(x::$VT) = $VT(@builtin_ccall("lgamma", $LVT, ($LVT,), x.data))

        @device_override @inline SpecialFunctions.gamma(x::$VT) = $VT(@builtin_ccall("tgamma", $LVT, ($LVT,), x.data))
    end
end

# see SPIRVIntrinsicsSpecialFunctionsExt
__init__() = append!(SPIRVIntrinsics.known_intrinsics, known_intrinsics)

end # module
