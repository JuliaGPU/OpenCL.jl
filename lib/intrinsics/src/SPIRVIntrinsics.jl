module SPIRVIntrinsics

using LLVM, LLVM.Interop
using Core: LLVMPtr

import ExprTools

import SpecialFunctions

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

end
