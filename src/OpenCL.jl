module OpenCL

using GPUCompiler
import GPUToolbox
import UnsafeAtomics
using LLVM, LLVM.IR, LLVM.Build, LLVM.Interop
using SPIRV_LLVM_Backend_jll, SPIRV_Tools_jll, spirv2clc_jll
using Adapt
using Reexport
using GPUArrays
using Random
using Preferences
using PrecompileTools: @compile_workload

import KernelInterface

using Core: LLVMPtr

# the size of an element in an array, which for isbits unions includes padding to their
# alignment. on Julia 1.10, `Base.aligned_sizeof` isn't constant-folded in device code.
@static if VERSION < v"1.11"
    @generated aligned_sizeof(::Type{T}) where {T} = :($(Base.aligned_sizeof(T)))
else
    import Base: aligned_sizeof
end

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
include("compiler/capabilities.jl")
include("compiler/compilation.jl")
include("compiler/exceptions.jl")
include("compiler/execution.jl")
include("compiler/reflection.jl")
include("compiler/precompile.jl")

# integrations and specialized functionality
include("util.jl")
include("broadcast.jl")
include("gpuarrays.jl")
include("random.jl")

include("OpenCLKernels.jl")
import .OpenCLKernels: OpenCLBackend
export OpenCLBackend

# not exported, because `@sync` would clash with `Base.@sync`
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Expr(:public, :synchronize, Symbol("@sync")))
end

# displaying an array (in USM, the preferred memory back-end where supported), which the
# device-free workload can't do, compiles for 150 to 400 ms otherwise. the REPL displays on
# a `TTYTerminal`, which is part of Base on 1.12+.
let ios = @static isdefined(Base, :Terminals) ? (IOBuffer, Base.Terminals.TTYTerminal) :
                                               (IOBuffer,)
    for io in ios, N in (1, 2)
        precompile(Tuple{typeof(show), IOContext{io}, MIME"text/plain",
                         CLArray{Float32,N,cl.UnifiedDeviceMemory}})
    end
end

# freeing an array, which otherwise compiles in the first garbage collection that frees one
for M in (cl.UnifiedDeviceMemory, cl.SharedVirtualMemory, cl.Buffer)
    precompile(Tuple{typeof(free), Managed{M}})
end

end
