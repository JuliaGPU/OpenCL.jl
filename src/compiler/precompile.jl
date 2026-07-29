# Cache plumbing reached on every kernel launch. These specializations need no OpenCL device
# and keep the first launch from paying for GPUCompiler's generic results machinery itself.
let Job = OpenCLCompilerJob
    precompile(Tuple{typeof(compile_or_lookup), Job})
    precompile(Tuple{typeof(GPUCompiler.cached_results), Type{OpenCLResults}, Job})
end
