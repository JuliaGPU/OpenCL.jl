## gpucompiler interface

Base.@kwdef struct OpenCLCompilerParams <: AbstractCompilerParams
    sub_group_size::Int # Some devices support multiple sizes. This is used to force one when needed
end

const OpenCLCompilerConfig = CompilerConfig{SPIRVCompilerTarget, OpenCLCompilerParams}
const OpenCLCompilerJob = CompilerJob{SPIRVCompilerTarget,OpenCLCompilerParams}

"""
    OpenCLResults

Cached compilation results attached to each OpenCL `CodeInstance`. Fields are populated
through the compile pipeline: `bitcode` after LLVM codegen (for runtime functions, which
GPUCompiler links into the kernel module — see `GPUCompiler.bitcode`/`bitcode!`),
`obj` (SPIR-V bytes) + `entry` + `device_rng` after main codegen, and `kernels` after
the session-local link onto an OpenCL context. The first four are session-portable
(cached through precompilation); `kernels` is session-local.

`kernels` is a small linear cache of `(cl.Context, cl.Kernel)` pairs. The cache partition
already covers everything that affects codegen via `GPUCompiler.cache_owner`, so the only
runtime-visible dimension left is the OpenCL context that owns the linked `cl.Kernel`.
A linear scan with `===` is fastest in the common case (n=1) and stays cheap for the
rare workload that bounces between a handful of contexts on the same device.
"""
mutable struct OpenCLResults
    bitcode::Union{Nothing, Tuple{Bool, Vector{UInt8}}}  # (opaque_pointers, bytes)
    obj::Union{Nothing, Vector{UInt8}}                   # SPIR-V binary
    entry::Union{Nothing, String}
    device_rng::Bool
    kernels::Vector{Tuple{cl.Context, cl.Kernel}}        # session-local; linear-scanned
    OpenCLResults() = new(nothing, nothing, nothing, false, Tuple{cl.Context, cl.Kernel}[])
end

function GPUCompiler.bitcode(r::OpenCLResults, opaque_pointers::Bool)
    r.bitcode === nothing && return nothing
    stored, bytes = r.bitcode
    return stored === opaque_pointers ? bytes : nothing
end

function GPUCompiler.bitcode!(r::OpenCLResults, bytes::Vector{UInt8}, opaque_pointers::Bool)
    r.bitcode = (opaque_pointers, bytes)
    return nothing
end

GPUCompiler.runtime_module(::CompilerJob{<:Any,OpenCLCompilerParams}) = OpenCL

GPUCompiler.results_type(::OpenCLCompilerJob) = OpenCLResults

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

    # Set the subgroup size if supported
    sg_size = job.config.params.sub_group_size
    if sg_size >= 0
        metadata(entry)["intel_reqd_sub_group_size"] = MDNode([ConstantInt(Int32(sg_size))])
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

    # Set to -1 if specifying a subgroup size is not supported
    sub_group_size = if "cl_intel_required_subgroup_size" in dev.extensions
        cl.sub_group_size(dev)
    else
        -1
    end

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; supports_fp16, supports_fp64, validate=true, kwargs...)
    params = OpenCLCompilerParams(; sub_group_size)
    CompilerConfig(target, params; kernel, name, always_inline)
end

# run inference + LLVM codegen + SPIR-V emission. returns `(obj, entry, device_rng)`,
# all session-portable so they survive precompilation when stored on a cached `CodeInstance`.
const compilations = Threads.Atomic{Int}(0)
function compile_to_obj(@nospecialize(job::CompilerJob))
    compilations[] += 1

    JuliaContext() do ctx
        obj, meta = GPUCompiler.compile(:obj, job)
        entry = LLVM.name(meta.entry)
        device_rng = StringAttribute("julia.opencl.rng", "") in collect(function_attributes(meta.entry))
        (; obj, entry, device_rng)
    end
end

# link the SPIR-V bytes into a session-local `cl.Kernel` on the active context.
function link_kernel(obj::Vector{UInt8}, entry::String)
    prog = if "cl_khr_il_program" in cl.device().extensions
        cl.Program(; il=obj)
    else
        error("Your device does not support SPIR-V, which is currently required for native execution.")
        # XXX: kpet/spirv2clc#87, caused by KhronosGroup/SPIRV-LLVM-Translator#2029
        source = mktempdir() do dir
            il = joinpath(dir, "kernel.spv")
            write(il, obj)
            cmd = `spirv2clc $il`
            read(cmd, String)
        end
        cl.Program(; source)
    end
    cl.build!(prog)
    return cl.Kernel(prog, entry)
end
