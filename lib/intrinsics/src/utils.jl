const opencl_builtins = String["printf"]

# OpenCL functions need to be mangled according to the C++ Itanium spec. We implement a very
# limited version of that spec here, just enough to support OpenCL built-ins.
#
# This macro also keeps track of called builtins, generating `ccall("extern...", llvmcall)`
# expressions for them (so that we can exclude them during IR verification).
macro builtin_ccall(name, ret, argtypes, args...)
    @assert Meta.isexpr(argtypes, :tuple)
    argtypes = argtypes.args

    function mangle(T::Type)
        if T == Cint
            "i"
        elseif T == Cuint
            "j"
        elseif T == Clong
            "l"
        elseif T == Culong
            "m"
        elseif T == Clonglong
            "x"
        elseif T == Culonglong
            "y"
        elseif T == Cshort
            "s"
        elseif T == Cushort
            "t"
        elseif T == Cchar
            "c"
        elseif T == Cuchar
            "h"
        elseif T == Cfloat
            "f"
        elseif T == Cdouble
            "d"
        elseif T <: LLVMPtr
            elt, as = T.parameters

            # mangle address space
            ASstr = if as == AS.Global
                "CLglobal"
            #elseif as == AS.Global_device
            #    "CLdevice"
            #elseif as == AS.Global_host
            #    "CLhost"
            elseif as == AS.Local
                "CLlocal"
            elseif as == AS.Constant
                "CLconstant"
            elseif as == AS.Private
                "CLprivate"
            elseif as == AS.Generic
                "CLgeneric"
            else
                error("Unknown address space $AS")
            end

            # encode as vendor qualifier
            ASstr = "U" * string(length(ASstr)) * ASstr

            # XXX: where does the V come from?
            "P" * ASstr * "V" * mangle(elt)
        else
            error("Unknown type $T")
        end
    end

    # C++-style mangling; very limited to just support these intrinsics
    # TODO: generalize for use with other intrinsics? do we need to mangle those?
    mangled = "_Z$(length(name))$name"
    for t in argtypes
        # with `@eval @builtin_ccall`, we get actual types in the ast, otherwise symbols
        t = (isa(t, Symbol) || isa(t, Expr)) ? eval(t) : t
        mangled *= mangle(t)
    end

    push!(opencl_builtins, mangled)
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
