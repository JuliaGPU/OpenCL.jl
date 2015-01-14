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

macro return_nanny_event(evt, obj)
    quote
        try
            return NannyEvent($(esc(evt)), $(esc(obj)))
        catch err
            @check api.clReleaseEvent($(esc(evt)))
            throw(err)
        end
    end
end

macro int_info(what, cl_obj_id, cl_obj_info, ret_type)
    local clFunc = symbol(string("clGet$(what)Info"))
    quote
        local result = Array($(esc(ret_type)), 1)
        local err::CL_int
        err = $clFunc($(esc(cl_obj)), $(esc(cl_obj_info)),
                      sizeof($(esc(ret_type))), result, C_NULL)
        if err != CL_SUCCESS
            throw(CLError(err))
        end
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

function _version_test(qm, elem :: Symbol, ex :: Expr, version :: VersionNumber)
    @assert qm == :?
    @assert ex.head == :(:)
    @assert length(ex.args) == 2

    esc(quote
        if OpenCL.check_version($elem, $version)
            $(ex.args[1])
        else
            $(ex.args[2])
        end
    end)
end

macro min_v11(qm, elem, ex)
    _version_test(qm, elem, ex, v"1.1")
end

macro min_v12(qm, elem, ex)
    _version_test(qm, elem, ex, v"1.2")
end

macro min_v20(qm, elem, ex)
    _version_test(qm, elem, ex, v"2.0")
end
