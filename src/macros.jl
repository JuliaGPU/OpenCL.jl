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
        local err::CL_int
        err = $clfunc
        if err != CL_SUCCESS
            throw(CLError(err))
        end
        err
    end
end

macro check_release(clfunc)
    quote
        local err::CL_int
        err = $clfunc 
        if err != CL_SUCCESS
            error("release! $clfunc failed with code $(err[1]))")
        end
    end
end

macro ocl_object_equality(cl_object_type)
    @eval begin 
        Base.hash(x::$cl_object_type) = unsigned(pointer(x))
        Base.isequal(x1::$cl_object_type, x2::$cl_object_type) = Base.hash(x1) == Base.hash(x2)
    end
end

#TODO: these are just stubs for future expanded versions
macro ocl_v1_1_only(ex)
    quote
        $(esc(ex))
    end
end

macro ocl_v1_2_only(ex)
    quote
        $(esc(ex))
    end
end

macro return_event(evt)
    quote
        try
            return Event($(esc(evt)), retain=false)
        catch err
            @check api.clReleaseEvent($(esc(evt)))
            throw(err)
        end
    end 
end

#TODO:
macro int_info(what, arg1, arg2, ret_type)
    local clFunc = symbol(string("clGet$(what)Info"))
    quote
        local result = Array($ret_type, 1)
        @check api.$clFunc($arg1, $arg2, sizeof($ret_type), result, C_NULL)
        result[1] 
    end
end

macro vec_info(what, arg1, arg2, res_vec)
    local clFunc = symbol(string("api.clGet$(what)Info"))
    quote
        local size = Array(Csize_t, 1)
        @check clFunc($arg1, $arg2, 0, C_NULL, size)
        local n = size / sizeof(res_vec[1])
        resize!(res_vec, n)
        @check clFunc($arg1, $arg2, empty($res_vec) ? C_NULL : res_vec, size)
    end
end

macro str_info(what, arg1, arg2)
    local clFunc = symbol("api.clGet$(what)Info")
    quote
        local size = Array(Csize_t, 1)
        @check $(esc(clFunc))($(esc(arg1)), $(esc(arg2)), 0, C_NULL, size)
        local result = Array(CL_char, size[1])
        @check $(esc(clFunc))($(esc(arg1)), $(esc(arg2)), size[1], result, size)
        bytestring(convert(Ptr{CL_char}, result))
    end
end
