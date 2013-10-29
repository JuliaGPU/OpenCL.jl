abstract CLMemObject

@ocl_func(clReleaseMemObject, (CL_mem,))

#TODO: function get_info()
#TODO: function hostbuf()
#TODO: release()
#TODO: host_array()
#TODO: pointer

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)

function release!(m::CLMemObject)
    if m.ptr != C_NULL
        clReleaseMemObject(m.ptr)
        m.ptr = C_NULL
    end
end

