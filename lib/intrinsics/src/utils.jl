const known_intrinsics = String["printf"]

# OpenCL functions need to be mangled according to the C++ Itanium spec. We implement a very
# limited version of that spec here, just enough to support OpenCL built-ins.
#
# This macro also keeps track of called builtins, generating `ccall("extern...", llvmcall)`
# expressions for them (so that we can exclude them during IR verification).
macro builtin_ccall(name, ret, argtypes, args...)
    @assert Meta.isexpr(argtypes, :tuple)
    argtypes = argtypes.args

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

    # C++-style mangling; very limited to just support these intrinsics
    # TODO: generalize for use with other intrinsics? do we need to mangle those?
    mangled = "_Z$(length(name))$name"
    for t in argtypes
        # with `@eval @builtin_ccall`, we get actual types in the ast, otherwise symbols
        t = (isa(t, Symbol) || isa(t, Expr)) ? __module__.eval(t) : t
        mangled *= mangle(t)
    end

    push!(__module__.known_intrinsics, mangled)
    esc(quote
        @typed_ccall($mangled, llvmcall, $ret, ($(argtypes...),), $(args...))
    end)
end


## device overrides

# local method table for device functions
Base.Experimental.@MethodTable(method_table)

macro device_override(ex)
    esc(quote
        Base.Experimental.@overlay($method_table, $ex)
    end)
end

macro device_function(ex)
    ex = macroexpand(__module__, ex)
    def = ExprTools.splitdef(ex)

    # generate a function that errors
    def[:body] = quote
        error("This function is not intended for use on the CPU")
    end

    esc(quote
        $(ExprTools.combinedef(def))
        @device_override $ex
    end)
end
