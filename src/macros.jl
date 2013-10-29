macro ocl_call(func, arg_types, args...)
    quote
        _err = ccall(($func, libopencl), CL_int, $arg_types, $(args...))
        if _err != CL_SUCCESS
            error("CL ERROR: $func")
        end
    end
end

macro ocl_call2(func, ret_type, arg_types, args...)
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

macro ocl_check(clfunc)
    quote
        _err = $clfunc
        if _err != CL_SUCCESS
            error("CL_ERROR: $func")
        end
    end
end
