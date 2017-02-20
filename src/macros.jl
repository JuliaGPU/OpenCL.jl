macro check(clfunc)
    quote
        local err::CL_int
        err = $(esc(clfunc))
        if err != CL_SUCCESS
            throw(CLError(err))
        end
        err
    end
end

macro check_release(clfunc)
    quote
        local err::CL_int
        err = $(esc(clfunc))
        if err != CL_SUCCESS
            error("release! $($(string(clfunc))) failed with code $err.")
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
        evt = $(esc(evt))
        try
            return Event(evt, retain=false)
        catch err
            @check api.clReleaseEvent(evt)
            throw(err)
        end
    end
end

macro return_nanny_event(evt, obj)
    quote
        evt = $(esc(evt))
        try
            return NannyEvent(evt, $(esc(obj)))
        catch err
            @check api.clReleaseEvent(evt)
            throw(err)
        end
    end
end

function _version_test(qm, elem, ex::Expr, version::VersionNumber)
    @assert qm == :?
    @assert ex.head == :(:)
    @assert length(ex.args) == 2

    quote
        if cl.check_version($(esc(elem)), $version)
            $(esc(ex.args[1]))
        else
            $(esc(ex.args[2]))
        end
    end
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
