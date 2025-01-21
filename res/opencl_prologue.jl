# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CLError(res))
end

function check(f)
    res = retry_reclaim(err -> err == CL_OUT_OF_RESOURCES ||
                               err == CL_MEM_OBJECT_ALLOCATION_FAILURE ||
                               err == CL_OUT_OF_HOST_MEMORY) do
        f()
    end

    if res != CL_SUCCESS
        throw_api_error(res)
    end

    return
end

macro CL_MAKE_VERSION(major, minor, patch)
    quote
        VersionNumber($major, $minor, $patch)
    end
end
