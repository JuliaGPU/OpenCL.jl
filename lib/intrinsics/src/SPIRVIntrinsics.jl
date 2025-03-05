module SPIRVIntrinsics

using LLVM, LLVM.Interop
using Core: LLVMPtr

import ExprTools

import SpecialFunctions

# helper type for writing UInt32/Int32 literals
# TODO: upstream this
struct Literal{T} end
Base.:(*)(x::Number, ::Type{Literal{T}}) where {T} = T(x)
const i32 = Literal{Int32}
const u32 = Literal{UInt32}

include("pointer.jl")
include("utils.jl")

# OpenCL intrinsics
#
# we currently don't implement SPIR-V intrinsics directly, but rely on
# the SPIR-V to LLVM translator supporting OpenCL intrinsics
include("work_item.jl")
include("synchronization.jl")
include("memory.jl")
include("printf.jl")
include("math.jl")
include("integer.jl")
include("atomic.jl")

# helper macro to import all names from this package, even non-exported ones.
macro import_all()
    code = quote end

    for name in names(SPIRVIntrinsics; all=true)
        # bring all the names of this module in scope
        name in (:SPIRVIntrinsics, :eval, :include) && continue
        startswith(string(name), "#") && continue
        push!(code.args, :(using .SPIRVIntrinsics: $name))
    end

    return code
end

# helper macro to re-export public names from this package
macro reexport_public()
    code = quote end

    for name in names(SPIRVIntrinsics)
        name == :SPIRVIntrinsics && continue
        push!(code.args, :(export $name))
    end

    return code
end

end
