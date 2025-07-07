## gpucompiler interface

struct OpenCLCompilerParams <: AbstractCompilerParams end
const OpenCLCompilerConfig = CompilerConfig{SPIRVCompilerTarget, OpenCLCompilerParams}
const OpenCLCompilerJob = CompilerJob{SPIRVCompilerTarget,OpenCLCompilerParams}

GPUCompiler.runtime_module(::CompilerJob{<:Any,OpenCLCompilerParams}) = OpenCL

GPUCompiler.method_table(::OpenCLCompilerJob) = method_table

# filter out OpenCL built-ins
# TODO: eagerly lower these using the translator API
GPUCompiler.isintrinsic(job::OpenCLCompilerJob, fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(fn)},
           job, fn) ||
    in(fn, opencl_builtins)


## compiler implementation (cache, configure, compile, and link)

# cache of compilation caches, per context
const _compiler_caches = Dict{cl.Context, Dict{Any, Any}}()
function compiler_cache(ctx::cl.Context)
    cache = get(_compiler_caches, ctx, nothing)
    if cache === nothing
        cache = Dict{Any, Any}()
        _compiler_caches[ctx] = cache
    end
    return cache
end

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, OpenCLCompilerConfig}()
function compiler_config(dev::cl.Device; kwargs...)
    h = hash(dev, hash(kwargs))
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(dev; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false, kwargs...)
    supports_fp16 = "cl_khr_fp16" in dev.extensions
    supports_fp64 = "cl_khr_fp64" in dev.extensions

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; supports_fp16, supports_fp64, validate=true, kwargs...)
    params = OpenCLCompilerParams()
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
const compilations = Threads.Atomic{Int}(0)
function compile(@nospecialize(job::CompilerJob))
    # TODO: this creates a context; cache those.
    obj, meta = JuliaContext() do ctx
        GPUCompiler.compile(:obj, job)
    end
    compilations[] += 1

    (obj, entry=LLVM.name(meta.entry))
end

function run_and_collect(cmd)
    stdout = Pipe()
    proc = run(pipeline(ignorestatus(cmd); stdout, stderr=stdout), wait=false)
    close(stdout.in)

    reader = Threads.@spawn String(read(stdout))
    Base.wait(proc)
    log = strip(fetch(reader))

    return proc, log
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    spirv_bitcode = compiled.obj
    clc_source = nothing

    prog = if "cl_khr_il_program" in cl.device().extensions
        cl.Program(; il=spirv_bitcode)
    else
        @warn """The current active OpenCL device '$(cl.device().name)' does not support IL programs.
                 Falling back to experimental SPIR-V to OpenCL C translation.""" maxlog=1 _id=Symbol(cl.device().name)
        spirv_path = tempname(cleanup=false) * ".spv"
        write(spirv_path, spirv_bitcode)
        proc, log = run_and_collect(`$(spirv2clc_jll.spirv2clc()) $spirv_path`)
        if !success(proc)
            msg = "Failed to translate SPIR-V to OpenCL C source code:\n$(log)"
            msg *= "\nIf you think this is a bug, please file an issue and attach $spirv_path"
            if parse(Bool, get(ENV, "BUILDKITE", "false"))
                run(`buildkite-agent artifact upload $spirv_path`)
            end
            error(msg)
        end
        rm(spirv_path)
        clc_source = strip(log)
        cl.Program(; source=clc_source)
    end

    try
        cl.build!(prog)
    catch e
        spirv_path = tempname(cleanup=false) * ".spv"
        write(spirv_path, spirv_bitcode)
        files = [spirv_path]
        if clc_source !== nothing
            clc_path = tempname(cleanup=false) * ".cl"
            write(clc_path, clc_source)
            push!(files, clc_path)
        end

        msg = "Failed to compile OpenCL program"
        msg *= "\nIf you think this is a bug, please file an issue and attach $(join(files, " and "))"
        if parse(Bool, get(ENV, "BUILDKITE", "false"))
            for file in files
                run(`buildkite-agent artifact upload $file`)
            end
        end
        error(msg)
    end
    cl.Kernel(prog, compiled.entry)
end
