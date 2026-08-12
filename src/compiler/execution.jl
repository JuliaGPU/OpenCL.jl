export @opencl, clfunction


## high-level @opencl interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline, :extensions, :backend, :validate, :sub_group_size]
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
                $kernel = $clfunction($kernel_f, $kernel_tt;
                                      source=$f_var, $(compiler_kwargs...))
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
    # memory objects to pass to `clSetKernelExecInfo` for indirect access
    indirect_memory::Vector{cl.AbstractMemory}
    # managed allocations whose ownership needs to be established at launch
    managed::Vector{Managed}
end

# when converting to pointers, we need to keep track of the underlying memory type
function Adapt.adapt_storage(to::KernelAdaptor, buf::cl.AbstractMemory)
    ptr = pointer(buf)
    push!(to.indirect_memory, buf)
    return ptr
end
function Adapt.adapt_storage(to::KernelAdaptor, arr::CLArray{T, N}) where {T, N}
    managed = arr.data[]
    push!(to.indirect_memory, managed.mem)
    push!(to.managed, managed)
    # note that conversion is pure: it does not migrate ownership of the allocation,
    # which only happens as part of the launch transaction (`take_ownership!`), under
    # the allocation lock and atomically with the enqueue of the kernel.
    ptr = convert(CLPtr{T}, managed.mem) + arr.offset
    return CLDeviceArray{T, N, AS.CrossWorkgroup}(
        size(arr), reinterpret(LLVMPtr{T, AS.CrossWorkgroup}, ptr),
        arr.maxsize - arr.offset)
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
kernel_convert(arg, indirect_memory::Vector{cl.AbstractMemory} = cl.AbstractMemory[],
               managed::Vector{Managed} = Managed[]) =
    adapt(KernelAdaptor(indirect_memory, managed), arg)

## abstract kernel functionality

abstract type AbstractKernel{F, TT} end

pass_arg(@nospecialize dt) = !(isghosttype(dt) || Core.Compiler.isconstType(dt))

@inline @generated function (kernel::AbstractKernel{F,TT})(args...;
                                                           call_kwargs...) where {F,TT}
    sig = Tuple{F, TT.parameters...}    # Base.signature_type with a function type
    args = (:(kernel_convert(source, indirect_memory, managed)),
            (:(kernel_convert(args[$i], indirect_memory, managed)) for i in 1:length(args))...)

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

    # convert arguments before locking (conversion is pure, and collects the managed
    # allocations to lock), then perform the launch transaction: with all participating
    # allocations locked in stable order, migrate their queue ownership and enqueue the
    # kernel. the source callable and arguments are preserved throughout, keeping the
    # converted pointers valid: neither the converted values nor the collected `Managed`
    # objects retain the `DataRef` ownership of their allocation.
    converted = [gensym(:arg) for _ in call_args]
    conversions = [:($(converted[i]) = $(call_args[i])) for i in 1:length(call_args)]
    quote
        indirect_memory = cl.AbstractMemory[]
        managed = Managed[]
        source = kernel.source
        GC.@preserve source args begin
            $(conversions...)
            locked = lock_managed(managed)
            try
                foreach(take_ownership!, locked)
                cl.call(kernel.fun, $(converted...);
                        indirect_memory, kernel.rng_state, call_kwargs...)
            finally
                unlock_managed(locked)
            end
        end
    end
end



## host-side kernels

struct HostKernel{F,S,TT} <: AbstractKernel{F,TT}
    f::F
    source::S
    fun::cl.Kernel
    rng_state::Bool
end


## host-side API

const clfunction_lock = ReentrantLock()

function clfunction(f::F, tt::TT=Tuple{}; source=f, kwargs...) where {F,TT}
    Base.@lock clfunction_lock begin
        config = compiler_config(cl.device(); kwargs...)::OpenCLCompilerConfig
        mi = methodinstance(F, tt)
        job = CompilerJob(mi, config)

        res = compile_or_lookup(job)::OpenCLResults

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
            kernel = link_kernel(job, res.obj::Vector{UInt8}, res.entry::String)
            # Don't cache session-local kernel handles while precompiling: the
            # results struct is serialized into the package image along with its
            # CodeInstance, and the handles would come back dangling.
            if ccall(:jl_generating_output, Cint, ()) != 1
                push!(res.kernels, (ctx, kernel))
            end
        end

        HostKernel{F,typeof(source),tt}(f, source, kernel, res.device_rng)
    end
end

# Look up cached compile artifacts for `job`, compiling on miss. Storage is managed
# by `GPUCompiler.cached_results` (Julia's integrated code cache on 1.11+, which also
# persists artifacts through precompilation; a session-local store on 1.10).
#
# `obj === nothing` identifies an `OpenCLResults` that hasn't been compiled yet. The
# `compile_hook` check additionally forces the compile path so reflection-style
# consumers (`@device_code_*`) observe the compilation even on a cache hit.
function compile_or_lookup(@nospecialize(job::CompilerJob))::OpenCLResults
    res = GPUCompiler.cached_results(OpenCLResults, job)
    if res === nothing || res.obj === nothing || GPUCompiler.compile_hook[] !== nothing
        compiled = compile_to_obj(job)
        res = @something res GPUCompiler.cached_results(OpenCLResults, job)
        res.obj = compiled.obj
        res.entry = compiled.entry
        res.device_rng = compiled.device_rng
    end
    return res
end
