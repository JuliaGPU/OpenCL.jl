module SPIRVIntrinsicsSpecialFunctionsExt

using SPIRVIntrinsics
using SPIRVIntrinsics: @device_override, @builtin_ccall, @typed_ccall, method_table
import SpecialFunctions

const known_intrinsics = String[]

for gentype in SPIRVIntrinsics.generic_types
    @eval begin
        @device_override SpecialFunctions.erfc(x::$gentype) = @builtin_ccall("erfc", $gentype, ($gentype,), x)
        @device_override SpecialFunctions.erf(x::$gentype) = @builtin_ccall("erf", $gentype, ($gentype,), x)

        @device_override SpecialFunctions.loggamma(x::$gentype) = @builtin_ccall("lgamma", $gentype, ($gentype,), x)

        @device_override SpecialFunctions.gamma(x::$gentype) = @builtin_ccall("tgamma", $gentype, ($gentype,), x)
    end
end

# compilers only consult the package's list of builtins, and additions to it made while
# precompiling this extension are not saved, so register them when the extension loads
__init__() = append!(SPIRVIntrinsics.known_intrinsics, known_intrinsics)

end # module
