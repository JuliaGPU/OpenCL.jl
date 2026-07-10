## gpucompiler interface

Base.@kwdef struct OpenCLCompilerParams <: AbstractCompilerParams
    # request a fixed sub-group width via `intel_reqd_sub_group_size`
    sub_group_size::Union{Nothing,Int} = nothing
    # optional features the target device supports, exposed to kernels via `has_feature`
    features::FeatureSet = zero(FeatureSet)
    # requested backend policy; kept here so the cache keys on it (the backends share a context)
    program_backend::Symbol = :auto
end

const OpenCLCompilerConfig = CompilerConfig{SPIRVCompilerTarget, OpenCLCompilerParams}
const OpenCLCompilerJob = CompilerJob{SPIRVCompilerTarget,OpenCLCompilerParams}

"""
    OpenCLResults

Cached compilation results for an OpenCL kernel job, managed by
`GPUCompiler.cached_results`. Fields are populated through the compile pipeline:
`obj` (SPIR-V bytes) + `entry` + `device_rng` after codegen, and `kernels` after the
session-local link onto an OpenCL context. The first three are session-portable
(cached through precompilation, except when GPUCompiler marks the job
session-dependent and wipes its entries before image serialization); `kernels` is
session-local and never populated during precompilation. `obj === nothing`
identifies a job that has not been compiled yet.

`kernels` is a small linear cache of `(cl.Context, cl.Kernel)` pairs. The cache partition
already covers everything that affects codegen via `GPUCompiler.cache_owner`, so the only
runtime-visible dimension left is the OpenCL context that owns the linked `cl.Kernel`.
A linear scan with `===` is fastest in the common case (n=1) and stays cheap for the
rare workload that bounces between a handful of contexts on the same device.
"""
mutable struct OpenCLResults
    obj::Union{Nothing, Vector{UInt8}}                   # SPIR-V binary
    entry::Union{Nothing, String}
    device_rng::Bool
    kernels::Vector{Tuple{cl.Context, cl.Kernel}}        # session-local; linear-scanned
    OpenCLResults() = new(nothing, nothing, false, Tuple{cl.Context, cl.Kernel}[])
end

GPUCompiler.runtime_module(::CompilerJob{<:Any,OpenCLCompilerParams}) = OpenCL

GPUCompiler.method_table_view(job::OpenCLCompilerJob) =
    GPUCompiler.StackedMethodTable(job.world, method_table, SPIRVIntrinsics.method_table)

# filter out OpenCL built-ins
# TODO: eagerly lower these using the translator API
GPUCompiler.isintrinsic(job::OpenCLCompilerJob, fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{CompilerJob{SPIRVCompilerTarget}, typeof(fn)},
           job, fn) ||
    in(fn, known_intrinsics) ||
    let SPIRVIntrinsicsSIMDExt = Base.get_extension(SPIRVIntrinsics, :SPIRVIntrinsicsSIMDExt)
        SPIRVIntrinsicsSIMDExt !== nothing && in(fn, SPIRVIntrinsicsSIMDExt.known_intrinsics)
    end ||
    contains(fn, "__spirv_")

GPUCompiler.kernel_state_type(::OpenCLCompilerJob) = KernelState

function GPUCompiler.finish_module!(@nospecialize(job::OpenCLCompilerJob),
                                    mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_module!,
                   Tuple{CompilerJob{SPIRVCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)

    sg_size = job.config.params.sub_group_size
    if sg_size !== nothing
        metadata(entry)["intel_reqd_sub_group_size"] = MDNode([ConstantInt(Int32(sg_size))])
    end

    # materialize the feature bitset for `has_feature`, if the kernel referenced it. A constant
    # initializer plus private linkage lets the optimizer fold the loads and drop the global, so it
    # never reaches SPIR-V.
    if haskey(globals(mod), "__opencl_feature_bitset")
        gv = globals(mod)["__opencl_feature_bitset"]
        initializer!(gv, ConstantInt(LLVM.Int64Type(), job.config.params.features))
        linkage!(gv, LLVM.API.LLVMPrivateLinkage)
        constant!(gv, true)
    end

    # if this kernel uses our RNG, we should prime the shared state.
    # XXX: these transformations should really happen at the Julia IR level...
    if haskey(functions(mod), "julia.opencl.random_keys") && job.config.kernel
        # insert call to `initialize_rng_state`
        f = initialize_rng_state
        ft = typeof(f)
        tt = Tuple{}

        # create a deferred compilation job for `initialize_rng_state`
        src = methodinstance(ft, tt, GPUCompiler.tls_world_age())
        cfg = CompilerConfig(job.config; kernel=false, name=nothing)
        job = CompilerJob(src, cfg, job.world)
        id = length(GPUCompiler.deferred_codegen_jobs) + 1
        GPUCompiler.deferred_codegen_jobs[id] = job

        # generate IR for calls to `deferred_codegen` and the resulting function pointer
        top_bb = first(blocks(entry))
        bb = BasicBlock(top_bb, "initialize_rng")
        @dispose builder=IRBuilder() begin
            position!(builder, bb)
            subprogram = LLVM.subprogram(entry)
            if subprogram !== nothing
                loc = DILocation(0, 0, subprogram)
                debuglocation!(builder, loc)
            end
            debuglocation!(builder, first(instructions(top_bb)))

            # call the `deferred_codegen` marker function
            T_ptr = if LLVM.version() >= v"17"
                LLVM.PointerType()
            elseif VERSION >= v"1.12.0-DEV.225"
                LLVM.PointerType(LLVM.Int8Type())
            else
                LLVM.Int64Type()
            end
            T_id = convert(LLVMType, Int)
            deferred_codegen_ft = LLVM.FunctionType(T_ptr, [T_id])
            deferred_codegen = if haskey(functions(mod), "deferred_codegen")
                functions(mod)["deferred_codegen"]
            else
                LLVM.Function(mod, "deferred_codegen", deferred_codegen_ft)
            end
            fptr = call!(builder, deferred_codegen_ft, deferred_codegen, [ConstantInt(id)])

            # call the `initialize_rng_state` function
            rt = Core.Compiler.return_type(f, tt)
            llvm_rt = convert(LLVMType, rt)
            llvm_ft = LLVM.FunctionType(llvm_rt)
            fptr = inttoptr!(builder, fptr, LLVM.PointerType(llvm_ft))
            call!(builder, llvm_ft, fptr)
            br!(builder, top_bb)

            # note the use of the device-side RNG in this kernel
            push!(function_attributes(entry), StringAttribute("julia.opencl.rng", ""))
        end

        # XXX: put some of the above behind GPUCompiler abstractions
        #      (e.g., a compile-time version of `deferred_codegen`)
    end
    return entry
end

function GPUCompiler.finish_linked_module!(@nospecialize(job::OpenCLCompilerJob), mod::LLVM.Module)
    for f in GPUCompiler.kernels(mod)
        kernel_intrinsics = Dict(
            "julia.opencl.random_keys" => (; name = "random_keys", typ = LLVMPtr{UInt32, AS.Workgroup}),
            "julia.opencl.random_counters" => (; name = "random_counters", typ = LLVMPtr{UInt32, AS.Workgroup}),
        )
        GPUCompiler.add_input_arguments!(job, mod, f, kernel_intrinsics)
    end
    return
end

## compiler implementation (configure, compile, and link)

# cache of compiler configurations, per device (but additionally configurable via kwargs)
const _toolchain = Ref{Any}()
const _compiler_configs = Dict{UInt, OpenCLCompilerConfig}()
function compiler_config(dev::cl.Device; kwargs...)
    # key on the policy, not the resolved backend: resolving queries the device, so defer it to link
    backend = program_backend()
    h = hash(dev, hash(backend, hash(kwargs)))
    config = get(_compiler_configs, h, nothing)
    if config === nothing
        config = _compiler_config(dev, backend; kwargs...)
        _compiler_configs[h] = config
    end
    return config
end
@inline _sub_group_size(dev) = "cl_intel_required_subgroup_size" in dev.extensions ? cl.sub_group_size(dev) : nothing

const SPIRV_VERSION = v"1.4"

@noinline function _compiler_config(dev, backend; kernel=true, name=nothing, always_inline=false,
                                     sub_group_size::Union{Nothing,Int}=_sub_group_size(dev),
                                     extensions::AbstractVector{<:AbstractString}=String[], kwargs...)
    supports_fp16 = "cl_khr_fp16" in dev.extensions
    supports_fp64 = "cl_khr_fp64" in dev.extensions

    if sub_group_size !== nothing && !("cl_intel_required_subgroup_size" in dev.extensions)
        error("Device does not support cl_intel_required_subgroup_size")
    end

    spirv_ext = join(("+$ext" for ext in extensions), ",")

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; version=SPIRV_VERSION, supports_fp16, supports_fp64,
                                   validate=true, extensions=spirv_ext, kwargs...)
    params = OpenCLCompilerParams(; sub_group_size, features=device_features(dev),
                                    program_backend=backend)
    CompilerConfig(target, params; kernel, name, always_inline)
end

# run inference + LLVM codegen + SPIR-V emission. returns `(obj, entry, device_rng)`,
# all session-portable so they survive precompilation when stored on a cached `CodeInstance`.
const compilations = Threads.Atomic{Int}(0)
function compile_to_obj(@nospecialize(job::CompilerJob))
    Threads.atomic_add!(compilations, 1)

    JuliaContext() do ctx
        obj, meta = GPUCompiler.compile(:obj, job)
        entry = LLVM.name(meta.entry)
        device_rng = StringAttribute("julia.opencl.rng", "") in collect(function_attributes(meta.entry))
        (; obj, entry, device_rng)
    end
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

# How kernels are fed to the driver:
# - `:auto`   — SPIR-V (clCreateProgramWithIL) when the device supports IL programs, otherwise
#               OpenCL C source translated from the SPIR-V via spirv2clc.
# - `:spirv`  — always SPIR-V IL; error if the device doesn't support it.
# - `:opencl` — always OpenCL C source. Useful to exercise pocl's source path on a device that
#               also supports IL.
# Task-local, like `cl.device`/`cl.queue`; `resolve_program_backend` maps it to a concrete backend.

"""
    program_backend() -> Symbol

The requested program backend for the current task (`:auto`, `:spirv`, or `:opencl`).
"""
program_backend() = get(task_local_storage(), :CLProgramBackend, :auto)::Symbol

"""
    program_backend!(mode::Symbol)
    program_backend!(f::Function, mode::Symbol)

Select how kernels are fed to the driver for the current task: `:auto` (default), `:spirv` (SPIR-V
IL), or `:opencl` (OpenCL C source). The second form applies `mode` only for the duration of `f`.
"""
function program_backend!(mode::Symbol)
    mode in (:auto, :spirv, :opencl) ||
        throw(ArgumentError("invalid program backend $mode (expected :auto, :spirv or :opencl)"))
    task_local_storage(:CLProgramBackend, mode)
    return mode
end

function program_backend!(f::Base.Callable, mode::Symbol)
    old = program_backend()
    program_backend!(mode)
    try
        f()
    finally
        program_backend!(old)
    end
end

# Map the policy to a concrete backend for `dev`. Done in `link`, off the per-launch path.
function resolve_program_backend(dev::cl.Device, mode::Symbol = program_backend())
    mode === :opencl && return :opencl
    mode in (:auto, :spirv) ||
        throw(ArgumentError("invalid program backend $mode (expected :auto, :spirv or :opencl)"))
    if "cl_khr_il_program" in dev.extensions
        return :spirv
    elseif mode === :spirv
        error("Device '$(dev.name)' does not support SPIR-V IL programs")
    else  # :auto
        @warn "Device '$(dev.name)' lacks IL support; using OpenCL C translation." maxlog=1 _id=Symbol(dev.name)
        return :opencl
    end
end

# Dump compilation artifacts (`extension => data` pairs sharing one random base name, e.g.
# `dump_artifacts(".spv" => spv, ".cl" => src)`) to a directory for later inspection. The
# directory is `JULIA_OPENCL_DUMP_DIR` if set, else `$RUNNER_TEMP/opencl-compilation-dumps` on
# GitHub Actions, else `tempdir()`. On CI the files are surfaced as downloadable artifacts
# (`buildkite-agent artifact upload`, or a GitHub Actions `::notice`). Returns the written paths.
function dump_artifacts(artifacts::Pair{String}...)
    on_github = get(ENV, "GITHUB_ACTIONS", "false") == "true"
    dir = if haskey(ENV, "JULIA_OPENCL_DUMP_DIR")
        mkpath(ENV["JULIA_OPENCL_DUMP_DIR"])
    elseif on_github
        mkpath(joinpath(get(ENV, "RUNNER_TEMP", tempdir()), "opencl-compilation-dumps"))
    else
        tempdir()
    end
    stem = tempname(dir; cleanup=false)

    paths = String[]
    for (ext, data) in artifacts
        path = stem * ext
        write(path, data)
        push!(paths, path)
    end

    if parse(Bool, get(ENV, "BUILDKITE", "false"))
        for path in paths
            run(`buildkite-agent artifact upload $path`)
        end
    elseif on_github
        println("::notice title=OpenCL compilation dump::wrote $(join(basename.(paths), ", ")) to $dir")
    end

    return paths
end

# link the SPIR-V bytes into a session-local `cl.Kernel` on the active context.
function link_kernel(@nospecialize(job::CompilerJob), obj::Vector{UInt8}, entry::String)
    spirv_bitcode = obj
    clc_source = nothing
    build_options = ""

    dev = cl.device()
    backend = resolve_program_backend(dev, job.config.params.program_backend)

    prog = if backend === :spirv
        cl.Program(; il=spirv_bitcode)
    else
        # Target the device's highest OpenCL C version
        clc_version = max_opencl_c_version(dev)
        cl_std = "CL$(clc_version.major).$(clc_version.minor)"

        # Be consistent with the SPIR-V version we generated code for
        spv = job.config.target.version
        spirv_version = "$(spv.major).$(spv.minor)"

        spirv_path = tempname(cleanup=false) * ".spv"
        write(spirv_path, spirv_bitcode)
        proc, log = run_and_collect(`$(spirv2clc_jll.spirv2clc()) --spirv-version=$spirv_version --cl-std=$cl_std $spirv_path`)
        rm(spirv_path)
        if !success(proc)
            spv_file, = dump_artifacts(".spv" => spirv_bitcode)
            error("Failed to translate SPIR-V to OpenCL C source code:\n$(log)\n" *
                  "If you think this is a bug, please file an issue and attach $spv_file")
        end

        clc_source = strip(log)
        build_options = "-cl-std=$cl_std"
        cl.Program(; source=clc_source)
    end

    # optionally dump the artifacts of every kernel (e.g. to debug a runtime miscompile)
    if haskey(ENV, "JULIA_OPENCL_DUMP_DIR")
        clc_source === nothing ? dump_artifacts(".spv" => spirv_bitcode) :
                                 dump_artifacts(".spv" => spirv_bitcode, ".cl" => clc_source)
    end

    try
        cl.build!(prog; options=build_options)
    catch e
        files = clc_source === nothing ?
            dump_artifacts(".spv" => spirv_bitcode) :
            dump_artifacts(".spv" => spirv_bitcode, ".cl" => clc_source)

        # `build!` already renders the source and build log; keep that and point at the artifacts
        msg = sprint(showerror, e)
        msg *= "\nIf you think this is a bug, please file an issue and attach $(join(files, " and "))"
        error(msg)
    end
    return cl.Kernel(prog, entry)
end
