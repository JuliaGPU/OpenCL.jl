module OpenCL

using GPUCompiler
using LLVM, LLVM.Interop
using SPIRV_LLVM_Translator_unified_jll
using Adapt
using Reexport

# library wrappers
include("../lib/CL.jl")
@reexport using .cl
export cl

# device functionality
include("device/runtime.jl")

# high-level functionality
include("util.jl")
include("array.jl")
include("compiler/compilation.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")

end
