## gpucompiler interface

Base.@kwdef struct OpenCLCompilerParams <: AbstractCompilerParams
    replace_copysign_f16 = false
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
    contains(fn, "__spirv_")

function GPUCompiler.finish_ir!(job::OpenCLCompilerJob, mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(GPUCompiler.finish_ir!,
                   Tuple{CompilerJob{SPIRVCompilerTarget}, LLVM.Module, LLVM.Function},
                   job, mod, entry)

    # replace copysign.f16 intrinsic with manual implementation since pocl doesn't support it
    job.config.params.replace_copysign_f16 && replace_copysign_f16!(mod)

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
    replace_copysign_f16 = dev.platform.name == "Portable Computing Language"

    # create GPUCompiler objects
    target = SPIRVCompilerTarget(; supports_fp16, supports_fp64, validate=true, kwargs...)
    params = OpenCLCompilerParams(; replace_copysign_f16)
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

# replace llvm.copysign.f16 calls with manual implementation
function replace_copysign_f16!(mod::LLVM.Module)
    changed = false
    GPUCompiler.@tracepoint "replace copysign f16" begin

    # Find llvm.copysign.f16 intrinsic
    copysign_name = "llvm.copysign.f16"
    if haskey(functions(mod), copysign_name)
        copysign_fn = functions(mod)[copysign_name]

        # Process all uses of the intrinsic
        for use in uses(copysign_fn)
            call_inst = user(use)
            if isa(call_inst, LLVM.CallInst)
                @dispose builder=IRBuilder() begin
                    # Position builder before the call
                    position!(builder, call_inst)

                    # Get operands (x and y)
                    x = operands(call_inst)[1]  # magnitude
                    y = operands(call_inst)[2]  # sign source

                    # Create the replacement implementation
                    i16_type = LLVM.IntType(16)

                    # Bitcast half values to i16
                    x_bits = bitcast!(builder, x, i16_type)
                    y_bits = bitcast!(builder, y, i16_type)

                    # XOR the bit patterns and check if result is negative
                    xor_result = xor!(builder, y_bits, x_bits)
                    is_negative = icmp!(builder, LLVM.API.LLVMIntSLT, xor_result,
                                      ConstantInt(i16_type, 0))

                    # Create fneg of x
                    neg_x = fneg!(builder, x)

                    # Select between neg_x and x based on the sign test
                    result = select!(builder, is_negative, neg_x, x)

                    # Replace uses and erase the original call
                    replace_uses!(call_inst, result)
                    erase!(call_inst)
                    changed = true
                end
            end
        end

        # Remove the intrinsic declaration if no longer used
        if isempty(uses(copysign_fn))
            erase!(copysign_fn)
        end
    end

    end
    return changed
end
