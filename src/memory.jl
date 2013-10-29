abstract CLMemObject

@ocl_func(clReleaseMemObject, (CL_mem,))

function free!(m::CLMemObject)
    if m.ptr != C_NULL
        clReleaseMemObject(m.ptr)
        m.ptr = C_NULL
    end
end

