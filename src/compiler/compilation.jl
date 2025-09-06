## gpucompiler interface

struct OpenCLCompilerParams <: AbstractCompilerParams end
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
    contains(fn, "__spirv_")

GPUCompiler.kernel_state_type(job::OpenCLCompilerJob) = KernelState
GPUCompiler.additional_arg_types(job::OpenCLCompilerJob) = (; random_keys = LLVMPtr{UInt32, AS.Workgroup}, random_counters = LLVMPtr{UInt32, AS.Workgroup})

function GPUCompiler.finish_module!(@nospecialize(job::OpenCLCompilerJob),
                                    mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_module!,
                   Tuple{CompilerJob{SPIRVCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)

    # if this kernel uses our RNG, we should prime the shared state.
    # XXX: these transformations should really happen at the Julia IR level...
    if callconv(entry) == LLVM.API.LLVMSPIRKERNELCallConv && haskey(functions(mod), "julia.gpu.additional_arg_getter")
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
        end

        # XXX: put some of the above behind GPUCompiler abstractions
        #      (e.g., a compile-time version of `deferred_codegen`)

    end
    return entry
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

# link into an executable kernel
function link(@nospecialize(job::CompilerJob), compiled)
    prog = if "cl_khr_il_program" in cl.device().extensions
        cl.Program(; il=compiled.obj)
    else
        error("Your device does not support SPIR-V, which is currently required for native execution.")
        # XXX: kpet/spirv2clc#87, caused by KhronosGroup/SPIRV-LLVM-Translator#2029
        source = mktempdir() do dir
            il = joinpath(dir, "kernel.spv")
            write(il, compiled.obj)
            cmd = `spirv2clc $il`
            read(cmd, String)
        end
        cl.Program(; source)
    end
    cl.build!(prog)
    cl.Kernel(prog, compiled.entry)
end
