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
            api.clReleaseEvent(evt)
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
            api.clReleaseEvent(evt)
            throw(err)
        end
    end
end

function _version_test(qm, elem, ex::Expr, version::VersionNumber, name)
    Base.depwarn("`@$name? elem ex1 : ex2` is deprecated, use `$name(elem) ? ex1 : ex2` instead", Symbol("@", name))
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
    _version_test(qm, elem, ex, v"1.1", :min_v11)
end

macro min_v12(qm, elem, ex)
    _version_test(qm, elem, ex, v"1.2", :min_v12)
end

macro min_v20(qm, elem, ex)
    _version_test(qm, elem, ex, v"2.0", :min_v20)
end
