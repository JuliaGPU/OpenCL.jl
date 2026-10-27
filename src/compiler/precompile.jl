# Cache plumbing reached on every kernel launch. These specializations need no OpenCL device
# and keep the first launch from paying for GPUCompiler's generic results machinery itself.
let Job = OpenCLCompilerJob
    precompile(Tuple{typeof(compile_or_lookup), Job})
    precompile(Tuple{typeof(GPUCompiler.cached_results), Type{OpenCLResults}, Job})
end

# Compile a kernel to SPIR-V without a device, using PoCL's target. This saves the compiler
# pipeline (inference, LLVM code generation and SPIR-V translation) in the package image,
# reducing compilation on the first launch. Julia 1.10 compiles less on the first launch
# and gains little from this workload.
#
# The kernel uses raw pointers, which have no bounds checks: with `--check-bounds=yes` they
# would need the runtime library, which the workload doesn't build.
function precompile_kernel(ptr)
    i = get_global_id()
    unsafe_store!(ptr, fma(2f0, unsafe_load(ptr, i), 1f0), i)
    return
end
function precompile_job()
    target = SPIRVCompilerTarget(; version=SPIRV_VERSION, validate=true, backend=:llvm,
                                   driver=:pocl)
    # Skip the runtime library, as GPUCompiler's workload does, to avoid leaking GPU code
    # into the package image on Windows (JuliaGPU/GPUCompiler.jl#601).
    config = CompilerConfig(target, OpenCLCompilerParams(); kernel=true, name=nothing,
                            always_inline=false, libraries=false)
    tt = Tuple{LLVMPtr{Float32,AS.CrossWorkgroup}}
    return CompilerJob(methodinstance(typeof(precompile_kernel), tt), config)
end
@static if VERSION >= v"1.11" && llvm_to_spirv_backend === :llvm
    # a failing workload only costs latency, so it mustn't fail precompilation
    try
        @compile_workload compile_or_lookup(precompile_job())
    catch err
        @debug "OpenCL.jl's precompile workload failed" exception=(err, catch_backtrace())
    end
    compilations[] = 0
end
