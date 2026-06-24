## gpucompiler interface

Base.@kwdef struct OpenCLCompilerParams <: AbstractCompilerParams
    # request a fixed sub-group width via `intel_reqd_sub_group_size`
    sub_group_size::Union{Nothing,Int} = nothing
    # optional features the target device supports, exposed to kernels via `has_feature`
    features::FeatureSet = zero(FeatureSet)
end

const OpenCLCompilerConfig = CompilerConfig{SPIRVCompilerTarget, OpenCLCompilerParams}
const OpenCLCompilerJob = CompilerJob{SPIRVCompilerTarget,OpenCLCompilerParams}

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
@inline _sub_group_size(dev) = "cl_intel_required_subgroup_size" in dev.extensions ? cl.sub_group_size(dev) : nothing

const SPIRV_VERSION = v"1.4"

@noinline function _compiler_config(dev; kernel=true, name=nothing, always_inline=false,
                                     sub_group_size::Union{Nothing,Int}=_sub_group_size(dev), kwargs...)
    supports_fp16 = "cl_khr_fp16" in dev.extensions
    supports_fp64 = "cl_khr_fp64" in dev.extensions

    if sub_group_size !== nothing && !("cl_intel_required_subgroup_size" in dev.extensions)
        error("Device does not support cl_intel_required_subgroup_size")
    end

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; version=SPIRV_VERSION, supports_fp16, supports_fp64,
                                   validate=true, kwargs...)
    params = OpenCLCompilerParams(; sub_group_size, features=device_features(dev))
    CompilerConfig(target, params; kernel, name, always_inline)
end

# compile to executable machine code
const compilations = Threads.Atomic{Int}(0)
function compile(@nospecialize(job::CompilerJob))
    compilations[] += 1

    # TODO: this creates a context; cache those.
    obj, meta = JuliaContext() do ctx
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
const program_backend = Ref{Symbol}(:auto)

"""
    program_backend!(mode::Symbol)

Select how kernels are fed to the driver: `:auto` (default), `:spirv` (SPIR-V IL), or `:opencl`
(OpenCL C source).
"""
function program_backend!(mode::Symbol)
    mode in (:auto, :spirv, :opencl) ||
        throw(ArgumentError("invalid program backend $mode (expected :auto, :spirv or :opencl)"))
    program_backend[] = mode
    return mode
end

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    spirv_bitcode = compiled.obj
    clc_source = nothing
    build_options = ""

    dev = cl.device()
    il_supported = "cl_khr_il_program" in dev.extensions
    backend = program_backend[]
    if backend == :spirv && !il_supported
        error("Device '$(dev.name)' does not support SPIR-V IL programs, but program_backend is :spirv")
    end
    use_il = backend == :spirv || (backend == :auto && il_supported)

    prog = if use_il
        cl.Program(; il=spirv_bitcode)
    else
        if backend == :auto
            @warn """The current active OpenCL device '$(dev.name)' does not support IL programs.
                     Falling back to experimental SPIR-V to OpenCL C translation.""" maxlog=1 _id=Symbol(dev.name)
        end

        # Target the device's highest OpenCL C version
        clc_version = max_opencl_c_version(dev)
        cl_std = "CL$(clc_version.major).$(clc_version.minor)"

        # Be consistent with the SPIR-V version we generated code for
        spv = job.config.target.version
        spirv_version = "$(spv.major).$(spv.minor)"

        spirv_path = tempname(cleanup=false) * ".spv"
        write(spirv_path, spirv_bitcode)
        proc, log = run_and_collect(`$(spirv2clc_jll.spirv2clc()) --spirv-version=$spirv_version --cl-std=$cl_std $spirv_path`)
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
        build_options = "-cl-std=$cl_std"
        cl.Program(; source=clc_source)
    end

    try
        cl.build!(prog; options=build_options)
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
    (; kernel=cl.Kernel(prog, compiled.entry), compiled.device_rng)
end
