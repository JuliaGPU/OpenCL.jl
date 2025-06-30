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

## device overrides

# local method table for device functions
Base.Experimental.@MethodTable(method_table)

macro device_override(ex)
    esc(quote
        Base.Experimental.@overlay($method_table, $ex)
    end)
end

macro device_function(ex)
    ex = macroexpand(__module__, ex)
    def = ExprTools.splitdef(ex)

    # generate a function that errors
    def[:body] = quote
        error("This function is not intended for use on the CPU")
    end

    esc(quote
        $(ExprTools.combinedef(def))
        @device_override $ex
    end)
end


# device functionality
import SPIRVIntrinsics
SPIRVIntrinsics.@import_all
SPIRVIntrinsics.@reexport_public

const spirv_method_table = SPIRVIntrinsics.method_table

include("device/runtime.jl")
include("device/array.jl")
include("device/quirks.jl")

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
