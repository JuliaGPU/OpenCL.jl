export @opencl, clfunction


## high-level @opencl interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline]
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

@inline @generated function (kernel::AbstractKernel{F,TT})(args...;
                                                           call_kwargs...) where {F,TT}
    sig = Tuple{F, TT.parameters...}    # Base.signature_type with a function type
    args = (:(kernel.f), (:(kernel_convert(args[$i], indirect_memory)) for i in 1:length(args))...)

    # filter out ghost arguments that shouldn't be passed
    predicate = dt -> isghosttype(dt) || Core.Compiler.isconstType(dt)
    to_pass = map(!predicate, sig.parameters)
    call_t =                  Type[x[1] for x in zip(sig.parameters,  to_pass) if x[2]]
    call_args = Union{Expr,Symbol}[x[1] for x in zip(args, to_pass)            if x[2]]

    # replace non-isbits arguments (they should be unused, or compilation would have failed)
    for (i,dt) in enumerate(call_t)
        if !isbitstype(dt)
            call_t[i] = Ptr{Any}
            call_args[i] = :C_NULL
        end
    end

    # finalize types
    call_tt = Base.to_tuple_type(call_t)

    quote
        indirect_memory = cl.AbstractMemory[]
        clcall(kernel.fun, $call_tt, $(call_args...); indirect_memory, call_kwargs...)
    end
end



## host-side kernels

struct HostKernel{F,TT} <: AbstractKernel{F,TT}
    f::F
    fun::cl.Kernel
end


## host-side API

const clfunction_lock = ReentrantLock()

function clfunction(f::F, tt::TT=Tuple{}; kwargs...) where {F,TT}
    ctx = cl.context()
    dev = cl.device()

    Base.@lock clfunction_lock begin
        # compile the function
        cache = compiler_cache(ctx)
        source = methodinstance(F, tt)
        config = compiler_config(dev; kwargs...)::OpenCLCompilerConfig
        fun = GPUCompiler.cached_compilation(cache, source, config, compile, link)

        # create a callable object that captures the function instance. we don't need to think
        # about world age here, as GPUCompiler already does and will return a different object
        h = hash(fun, hash(f, hash(tt)))
        kernel = get(_kernel_instances, h, nothing)
        if kernel === nothing
            # create the kernel state object
            kernel = HostKernel{F,tt}(f, fun)
            _kernel_instances[h] = kernel
        end
        return kernel::HostKernel{F,tt}
    end
end

# cache of kernel instances
const _kernel_instances = Dict{UInt, Any}()
