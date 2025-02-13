# printf

# Formatted Output (B.17)

@generated function promote_c_argument(arg)
    # > When a function with a variable-length argument list is called, the variable
    # > arguments are passed using C's old ``default argument promotions.'' These say that
    # > types char and short int are automatically promoted to int, and type float is
    # > automatically promoted to double. Therefore, varargs functions will never receive
    # > arguments of type char, short int, or float.

    if arg == Cchar || arg == Cshort || arg == Cuchar || arg == Cushort
        return :(Cint(arg))
    elseif arg == Cfloat
        return :(Cdouble(arg))
    else
        return :(arg)
    end
end

macro printf(fmt::String, args...)
    fmt_val = Val(Symbol(fmt))

    return :(emit_printf($fmt_val, $(map(arg -> :(promote_c_argument($arg)), esc.(args))...)))
end

@generated function emit_printf(::Val{fmt}, argspec...) where {fmt}
    arg_exprs = [:( argspec[$i] ) for i in 1:length(argspec)]
    arg_types = [argspec...]

    Context() do ctx
        T_void = LLVM.VoidType()
        T_int32 = LLVM.Int32Type()
        T_pint8 = LLVM.PointerType(LLVM.Int8Type(), AS.UniformConstant)

        # create functions
        param_types = LLVMType[convert(LLVMType, typ) for typ in arg_types]
        llvm_f, _ = create_function(T_int32, param_types)
        mod = LLVM.parent(llvm_f)

        IRBuilder() do builder
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            # `printf` needs to be invoked very specifically, e.g., the format string needs
            # to be a pointer to a string, and arguments need to match exactly what is
            # expected by the format string, so we cannot rely on how the arguments to this
            # function have been passed in (by `llvmcall`).
            T_actual_args = LLVMType[]
            actual_args = LLVM.Value[]
            for (_, (arg, argtyp)) in enumerate(zip(parameters(llvm_f), arg_types))
                if argtyp <: LLVMPtr
                    # passed as i8*
                    T,AS = argtyp.parameters
                    actual_typ = LLVM.PointerType(convert(LLVMType, T), AS)
                    actual_arg = bitcast!(builder, arg, actual_typ)
                elseif argtyp <: Ptr
                    T = eltype(argtyp)
                    if T === Nothing
                        T = Int8
                    end
                    actual_typ = LLVM.PointerType(convert(LLVMType, T))
                    actual_arg = if value_type(arg) isa LLVM.PointerType
                        # passed as i8* or ptr
                        bitcast!(builder, arg, actual_typ)
                    else
                        # passed as i64
                        inttoptr!(builder, arg, actual_typ)
                    end
                elseif argtyp <: Bool
                    # passed as i8
                    T = eltype(argtyp)
                    actual_typ = LLVM.Int1Type()
                    actual_arg = trunc!(builder, arg, actual_typ)
                else
                    actual_typ = convert(LLVMType, argtyp)
                    actual_arg = arg
                end
                push!(T_actual_args, actual_typ)
                push!(actual_args, actual_arg)
            end

            # `printf` needs to be invoked very specifically, e.g., the format string needs
            # to be a pointer to a string, and arguments need to match exactly what is
            # expected by the format string, so we cannot rely on how the arguments to this
            # function have been passed in (by `llvmcall`).
            T_actual_args = LLVMType[]
            actual_args = LLVM.Value[]
            for (_, (arg, argtyp)) in enumerate(zip(parameters(llvm_f), arg_types))
                if argtyp <: LLVMPtr
                    # passed as i8*
                    T,AS = argtyp.parameters
                    actual_typ = LLVM.PointerType(convert(LLVMType, T), AS)
                    actual_arg = bitcast!(builder, arg, actual_typ)
                elseif argtyp <: Ptr
                    T = eltype(argtyp)
                    if T === Nothing
                        T = Int8
                    end
                    actual_typ = LLVM.PointerType(convert(LLVMType, T))
                    actual_arg = if value_type(arg) isa LLVM.PointerType
                        # passed as i8* or ptr
                        bitcast!(builder, arg, actual_typ)
                    else
                        # passed as i64
                        inttoptr!(builder, arg, actual_typ)
                    end
                elseif argtyp <: Bool
                    # passed as i8
                    T = eltype(argtyp)
                    actual_typ = LLVM.Int1Type()
                    actual_arg = trunc!(builder, arg, actual_typ)
                else
                    actual_typ = convert(LLVMType, argtyp)
                    actual_arg = arg
                end
                push!(T_actual_args, actual_typ)
                push!(actual_args, actual_arg)
            end

            str = globalstring_ptr!(builder, String(fmt); addrspace=AS.UniformConstant)

            # invoke printf and return
            printf_typ = LLVM.FunctionType(T_int32, [T_pint8]; vararg=true)
            printf = LLVM.Function(mod, "printf", printf_typ)
            push!(function_attributes(printf), EnumAttribute("nobuiltin"))
            chars = call!(builder, printf_typ, printf, [str, actual_args...])

            ret!(builder, chars)
        end

        call_function(llvm_f, Int32, Tuple{arg_types...}, arg_exprs...)
    end
end


## print-like functionality

# simple conversions, defining an expression and the resulting argument type. nothing fancy,
# `@print` pretty directly maps to `@printf`; we should just support `write(::IO)`.
const print_conversions = Dict(
    Float32     => (x->:(Float64($x)),             Float64),
    Ptr{<:Any}  => (x->:(convert(Ptr{Cvoid}, $x)), Ptr{Cvoid}),
    Bool        => (x->:(Int32($x)),               Int32),
)

# format specifiers
const print_specifiers = Dict(
    # integers
    Int16       => "%hd",
    Int32       => "%d",
    Int64       => Sys.iswindows() ? "%lld" : "%ld",
    UInt16      => "%hu",
    UInt32      => "%u",
    UInt64      => Sys.iswindows() ? "%llu" : "%lu",

    # floating-point
    Float64     => "%f",

    # other
    Cchar       => "%c",
    Ptr{Cvoid}  => "%p",
)

@generated function _print(parts...)
    fmt = ""
    args = Expr[]

    for i in 1:length(parts)
        part = :(parts[$i])
        T = parts[i]

        # put literals directly in the format string
        if T <: Val
            fmt *= string(T.parameters[1])
            continue
        end

        # try to convert arguments if they are not supported directly
        if !haskey(print_specifiers, T)
            for Tmatch in keys(print_conversions)
                if T <: Tmatch
                    conv, T = print_conversions[Tmatch]
                    part = conv(part)
                    break
                end
            end
        end

        # render the argument
        if haskey(print_specifiers, T)
            fmt *= print_specifiers[T]
            push!(args, part)
        elseif T <: String
            @error("@print does not support non-literal strings")
        else
            @error("@print does not support values of type $T")
        end
    end

    quote
        Base.@_inline_meta
        @printf($fmt, $(args...))
    end
end

"""
    @print(xs...)
    @println(xs...)

Print a textual representation of values `xs` to standard output from the GPU. The
functionality builds on `@printf`, and is intended as a more use friendly alternative of
that API. However, that also means there's only limited support for argument types, handling
16/32/64 signed and unsigned integers, 32 and 64-bit floating point numbers, `Cchar`s and
pointers. For more complex output, use `@printf` directly.

Limited string interpolation is also possible:

```julia
    @print("Hello, World ", 42, "\\n")
    @print "Hello, World \$(42)\\n"
```
"""
macro print(parts...)
    args = Union{Val,Expr,Symbol}[]

    parts = [parts...]
    while true
        isempty(parts) && break

        part = popfirst!(parts)

        # handle string interpolation
        if isa(part, Expr) && part.head == :string
            parts = vcat(part.args, parts)
            continue
        end

        # expose literals to the generator by using Val types
        if isbits(part) # literal numbers, etc
            push!(args, Val(part))
        elseif isa(part, QuoteNode) # literal symbols
            push!(args, Val(part.value))
        elseif isa(part, String) # literal strings need to be interned
            push!(args, Val(Symbol(part)))
        else # actual values that will be passed to printf
            push!(args, part)
        end
    end

    quote
        _print($(map(esc, args)...))
    end
end

@doc (@doc @print) ->
macro println(parts...)
    esc(quote
        $SPIRVIntrinsics.@print($(parts...), "\n")
    end)
end

"""
    @show(ex)

GPU analog of `Base.@show`. It comes with the same type restrictions as [`@printf`](@ref).

```julia
@show threadIdx().x
```
"""
macro show(exs...)
    blk = Expr(:block)
    for ex in exs
        push!(blk.args, :($SPIRVIntrinsics.@println($(sprint(Base.show_unquoted,ex)*" = "),
                                                    begin local value = $(esc(ex)) end)))
    end
    isempty(exs) || push!(blk.args, :value)
    blk
end
