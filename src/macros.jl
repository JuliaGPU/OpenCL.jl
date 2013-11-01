macro ocl_call(func, arg_types, args...)
    quote
        _err = ccall(($func, libopencl), CL_int, $arg_types, $(args...))
        if _err != CL_SUCCESS
            error("CL ERROR: $func")
        end
    end
end

macro ocl_call(func, ret_type, arg_types, args...)
    quote
        ccall(($func, libopencl), $ret_type, $arg_types, $(args...))
    end
end

macro ocl_func(func, arg_types)
    local args_in = Symbol[symbol(string('a', i)) for i in 1:length(arg_types.args)]
    quote
        $(esc(func))($(args_in...)) = @ocl_call($(string(func)), $arg_types, $(args_in...))
    end
end

macro check(clfunc)
    quote
        _err = $clfunc
        if _err != CL_SUCCESS
            throw(CLError(err))
        end
    end
end

macro ocl_object_equality(cl_object_type)
    @eval begin 
        Base.hash(x::$cl_object_type) = unsigned(pointer(x))
        Base.isequal(x1::$cl_object_type, x2::$cl_object_type) = Base.hash(x1) == Base.hash(x2)
    end
end
