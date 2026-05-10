export @opencl, clfunction


## high-level @opencl interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline, :extensions, :backend, :validate]
const LAUNCH_KWARGS = [:global_size, :local_size, :queue]

macro opencl(ex...)
    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @opencl should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, COMPILER_KWARGS, LAUNCH_KWARGS)
    if !isempty(other_kwargs)
        key,val = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key,val = kwarg.args
        if key == :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @opencl should be a constant value"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@opencl with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym f_var kernel_f kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $kernel_convert($f_var)
                $kernel_args = map($kernel_convert, ($(var_exprs...),))
                $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                $kernel = $clfunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
         end)

    return esc(quote
        let
            $code
        end
    end)
end

## argument conversion

struct KernelAdaptor
    indirect_memory::Vector{cl.AbstractMemory}
end

# when converting to pointers, we need to keep track of the underlying memory type
function Adapt.adapt_storage(to::KernelAdaptor, buf::cl.AbstractMemory)
    ptr = pointer(buf)
    push!(to.indirect_memory, buf)
    return ptr
end
function Adapt.adapt_storage(to::KernelAdaptor, arr::CLArray{T, N}) where {T, N}
    push!(to.indirect_memory, arr.data[].mem)
    return Base.unsafe_convert(CLDeviceArray{T, N, AS.CrossWorkgroup}, arr)
end

# Base.RefValue isn't GPU compatible, so provide a compatible alternative
# TODO: port improvements from CUDA.jl
struct CLRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::CLRefValue) = r.x
Adapt.adapt_structure(to::KernelAdaptor, r::Base.RefValue) = CLRefValue(adapt(to, r[]))

# broadcast sometimes passes a ref(type), resulting in a GPU-incompatible DataType box.
# avoid that by using a special kind of ref that knows about the boxed type.
struct CLRefType{T} <: Ref{DataType} end
Base.getindex(r::CLRefType{T}) where T = T
Adapt.adapt_structure(to::KernelAdaptor, r::Base.RefValue{<:Union{DataType,Type}}) =
    CLRefType{r[]}()

# case where type is the function being broadcasted
Adapt.adapt_structure(to::KernelAdaptor,
                      bc::Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)

"""
    kernel_convert(x)

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `OpenCL.KernelAdaptor` type.
"""
kernel_convert(arg, indirect_memory::Vector{cl.AbstractMemory} = cl.AbstractMemory[]) =
    adapt(KernelAdaptor(indirect_memory), arg)

## abstract kernel functionality

abstract type AbstractKernel{F, TT} end

pass_arg(@nospecialize dt) = !(isghosttype(dt) || Core.Compiler.isconstType(dt))

@inline @generated function (kernel::AbstractKernel{F,TT})(args...;
                                                           call_kwargs...) where {F,TT}
    sig = Tuple{F, TT.parameters...}    # Base.signature_type with a function type
    args = (:(kernel.f), (:(kernel_convert(args[$i], indirect_memory)) for i in 1:length(args))...)

    # filter out ghost arguments that shouldn't be passed
    to_pass = map(pass_arg, sig.parameters)
    call_t =                  Type[x[1] for x in zip(sig.parameters,  to_pass) if x[2]]
    call_args = Union{Expr,Symbol}[x[1] for x in zip(args, to_pass)            if x[2]]

    # replace non-isbits arguments (they should be unused, or compilation would have failed)
    for (i,dt) in enumerate(call_t)
        if !isbitstype(dt)
            call_t[i] = Ptr{Any}
            call_args[i] = :C_NULL
        end
    end

    pushfirst!(call_t, KernelState)
    pushfirst!(call_args, :(KernelState(kernel.rng_state ? Base.rand(UInt32) : UInt32(0))))

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        indirect_memory = cl.AbstractMemory[]
        clcall(kernel.fun, $call_tt, $(call_args...); indirect_memory, kernel.rng_state, call_kwargs...)
    end
end



## host-side kernels

struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    f::F
    fun::cl.Kernel
    rng_state::Bool
end


## host-side API

const clfunction_lock = ReentrantLock()

function clfunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    Base.@lock clfunction_lock begin
        config = compiler_config(cl.device(); kwargs...)::OpenCLCompilerConfig
        source = methodinstance(F, tt)
        job = CompilerJob(source, config)
        cache = GPUCompiler.cache_view(job)

        ci, res = something(lookup(cache, source), compile_opencl!(cache, job))

        # Resolve the cl.Kernel for the active context. Linear scan over the
        # session-local cache; almost always n=1, so this is one `===` compare.
        ctx = cl.context()
        kernel = nothing
        @inbounds for (cached_ctx, cached_kernel) in res.kernels
            if cached_ctx === ctx
                kernel = cached_kernel
                break
            end
        end
        if kernel === nothing
            kernel = link_kernel(res.obj::Vector{UInt8}, res.entry::String)
            push!(res.kernels, (ctx, kernel))
        end

        h = hash(kernel, hash(f, hash(tt)))
        get!(_kernel_instances, h) do
            HostKernel{F,tt}(f, kernel, res.device_rng)
        end::HostKernel{F,tt}
    end
end

# Run inference and codegen for `job`, then populate the cached `OpenCLResults` with the
# session-portable artifacts. The `CodeInstance` is created during inference inside
# `GPUCompiler.compile` (which uses the same owner-partitioned `CacheView`), and gets a
# fresh `OpenCLResults()` attached via `@setup_caching`'s `finish!` hook.
function compile_opencl!(cache::CacheView, @nospecialize(job::CompilerJob))
    compiled = compile_to_obj(job)
    ci = get(cache, job.source, nothing)::Core.CodeInstance
    res = results(cache, ci)::OpenCLResults
    res.obj = compiled.obj
    res.entry = compiled.entry
    res.device_rng = compiled.device_rng
    return (ci, res)
end

# cache of kernel instances
const _kernel_instances = Dict{UInt, Any}()
