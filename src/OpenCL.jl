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
let
    # re-export functionality from SPIRVIntrinsics
    for name in names(SPIRVIntrinsics)
        name == :SPIRVIntrinsics && continue
        @eval export $name
    end

    # import all the others so that the user can refer to them through the OpenCL module
    for name in names(SPIRVIntrinsics; all=true)
        # bring all the names of this module in scope
        name in (:SPIRVIntrinsics, :eval, :include) && continue
        startswith(string(name), "#") && continue
        @eval begin
            using .SPIRVIntrinsics: $name
        end
    end
end
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
