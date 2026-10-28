const known_intrinsics = String["printf"]

# SPIR-V wrapper and OpenCL.std functions need Itanium C++ ABI names. We
# implement a very limited subset here, just enough to support these builtins.
function mangle(T::Type)
    if T == Int32
        "i"
    elseif T == UInt32
        "j"
    elseif T == Int64
        "l"
    elseif T == UInt64
        "m"
    elseif T == Int16
        "s"
    elseif T == UInt16
        "t"
    elseif T == Int8
        "c"
    elseif T == UInt8
        "h"
    elseif T == Float16
        "Dh"
    elseif T == Float32
        "f"
    elseif T == Float64
        "d"
    elseif T <: LLVMPtr
        elt, as = T.parameters
        (as == AS.Private ? "P" : "PU3AS$as") * "V" * mangle(elt)
    else
        error("Unknown type $T")
    end
end
mangle(::Type{NTuple{N, VecElement{T}}}) where {N, T} = "Dv$(N)_" * mangle(T)

# `@builtin_ccall` calls builtins by their mangled name. It also keeps track of called
# builtins, generating `ccall("extern...", llvmcall)` expressions for them (so that we can
# exclude them during IR verification).
#
# Builtins that all work-items of a (sub-)group have to execute together (barriers, sub-group
# shuffles, votes, reductions, ...) have to be called with `convergent = true`, see
# `convergent_builtin`.
macro builtin_ccall(name, ret, argtypes, args...)
    convergent = false
    if !isempty(args) && Meta.isexpr(args[end], :(=), 2) && args[end].args[1] == :convergent
        convergent = args[end].args[2]::Bool
        args = args[1:end-1]
    end
    @assert Meta.isexpr(argtypes, :tuple)
    argtypes = argtypes.args

    # C++-style mangling; very limited to just support these intrinsics
    # TODO: generalize for use with other intrinsics? do we need to mangle those?
    mangled = "_Z$(length(name))$name"
    for t in argtypes
        # with `@eval @builtin_ccall`, we get actual types in the ast, otherwise symbols
        t = (isa(t, Symbol) || isa(t, Expr)) ? __module__.eval(t) : t
        # `convergent_builtin` passes arguments as `llvmcall` lowers them, without the
        # `Bool` and pointer conversions of `@typed_ccall`
        convergent && (t <: Union{Bool, Ptr, LLVMPtr}) &&
            error("@builtin_ccall with `convergent = true` does not support arguments of type $t")
        mangled *= mangle(t)
    end

    push!(__module__.known_intrinsics, mangled)
    if convergent
        rettyp = (isa(ret, Symbol) || isa(ret, Expr)) ? __module__.eval(ret) : ret
        rettyp <: Union{Bool, Ptr, LLVMPtr} &&
            error("@builtin_ccall with `convergent = true` does not support return type $rettyp")
        @assert length(argtypes) == length(args)
        converted = [:(convert($T, $arg)) for (T, arg) in zip(argtypes, args)]
        esc(quote
            $convergent_builtin($(Val(Symbol(mangled))), $ret, $(converted...))
        end)
    else
        esc(quote
            @typed_ccall($mangled, llvmcall, $ret, ($(argtypes...),), $(args...))
        end)
    end
end

# call a builtin that all work-items of the (sub-)group have to execute together. its
# declaration is marked `convergent`, so that the optimizer doesn't make the call
# control-dependent on additional values, e.g., by duplicating it into the arms of a branch
# that computes its argument, which makes the work-items execute different calls.
@llvmgenerated builder function convergent_builtin(::Val{name}, ::Type{T},
                                                   args...)::T where {name, T}
    rt = T === Nothing ? LLVM.VoidType() : convert(LLVMType, T)
    ft = LLVM.FunctionType(rt, LLVMType[arg.value_type for arg in args])
    f = LLVM.Function(current_module(builder), String(name), ft)
    push!(f.function_attributes, EnumAttribute(:convergent))
    push!(f.function_attributes, EnumAttribute(:nounwind))
    rv = call!(builder, ft, f, collect(Value, args))
    T === Nothing ? nothing : rv
end


## device overrides

# local method table for device functions
Base.Experimental.@MethodTable(method_table)

const HAS_CONSISTENT_OVERLAY = VERSION >= v"1.12.0-DEV.745" || v"1.11-rc1" <= VERSION < v"1.12-"

# `method_table` resolves in the caller's module, so that back ends that use these macros
# define their overrides in their own method table
macro device_override(ex)
    esc(GPUToolbox.Overlays.device_override(__module__, GlobalRef(__module__, :method_table), ex))
end

macro device_function(ex)
    esc(GPUToolbox.Overlays.device_function(__module__, GlobalRef(__module__, :method_table), ex))
end
