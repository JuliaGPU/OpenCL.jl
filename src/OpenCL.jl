module OpenCL

using GPUCompiler
using LLVM, LLVM.Interop
using SPIRV_LLVM_Translator_unified_jll
using Adapt
using Reexport
using GPUArrays
using Random

using Core: LLVMPtr

# library wrappers
include("../lib/cl/CL.jl")
@reexport using .cl
export cl

# device functionality
include("device/runtime.jl")
import SPIRVIntrinsics
SPIRVIntrinsics.@import_all
SPIRVIntrinsics.@reexport_public
include("device/array.jl")
include("device/quirks.jl")

# compiler implementation
include("compiler/compilation.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")

# high-level functionality
include("util.jl")
include("array.jl")
include("mapreduce.jl")
include("gpuarrays.jl")

include("OpenCLKernels.jl")
import .OpenCLKernels: OpenCLBackend
export OpenCLBackend

end
