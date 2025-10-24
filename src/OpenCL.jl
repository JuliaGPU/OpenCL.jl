module OpenCL

using GPUCompiler
using LLVM, LLVM.Interop
using SPIRV_LLVM_Backend_jll, SPIRV_Tools_jll
using Adapt
using Reexport
using GPUArrays
using Random
using Preferences

using Core: LLVMPtr

# library wrappers
include("../lib/cl/CL.jl")
@reexport using .cl
export cl

# device functionality
import SPIRVIntrinsics
SPIRVIntrinsics.@import_all
SPIRVIntrinsics.@reexport_public
Base.Experimental.@MethodTable(method_table)
include("device/runtime.jl")
include("device/array.jl")
include("device/quirks.jl")
include("device/random.jl")

# high level implementation
include("memory.jl")
include("array.jl")

# compiler implementation
include("compiler/compilation.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")

# integrations and specialized functionality
include("util.jl")
include("broadcast.jl")
include("mapreduce.jl")
include("gpuarrays.jl")
include("random.jl")

include("OpenCLKernels.jl")
import .OpenCLKernels: OpenCLBackend
export OpenCLBackend
end
