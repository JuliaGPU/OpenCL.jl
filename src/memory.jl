abstract CLMemObject

#TODO: this should be implemented by all subtypes
#type MemObject <: CLMemory
#    valid :: Bool
#    ptr   :: CL_mem
#    hostbuf
#
#    function CLMemObject(mem_ptr::CL_mem; retain=true, hostbuf=nothing)
#        if retain
#            @check api.clRetainMemObject(mem_ptr)
#        end
#        m = new(true, mem_ptr, hostbuf)
#        finalizer(m, mem -> if mem.valid; release!(mem); end)
#        return m
#    end
#
#end

Base.pointer(mem::CLMemObject) = mem.id

Base.sizeof(mem::CL_mem) = begin
    val = Csize_t[0,]
    @check api.clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(Csize_t), val, C_NULL)
    return val[1]
end


function release!(mem::CLMemObject)
    if !mem.valid
        error("OpenCL.MemObject relase! error: trying to double unref mem object")
    end
    @check_release api.clReleaseMemObject(mem.id)
    mem.id = C_NULL
    mem.valid = false
end

context(mem::CLMemObject) = begin
    param = Array(CL_context, 1)
    @check api.clGetMemObjectInfo(mem.id, CL_MEM_CONTEXT, 
                                  sizeof(Csize_t), param_value, C_NULL)
    return Context(param[1])
end

macro memobj_property(func, cl_device_info, return_type)
    quote
        function $func(mem::CLMemObject)
            result = Array($return_type, 1)
            @check api.clGetMemObjectInfo(mem.id, $cl_device_info,
                                          sizeof($return_type), result, C_NULL)
            return result[1]
        end
    end
end

@memobj_property(mem_type,        CL_MEM_TYPE,            CL_mem_object_type)
@memobj_property(size,            CL_MEM_SIZE,            Csize_t)
@memobj_property(reference_count, CL_MEM_REFERENCE_COUNT, CL_uint)
@memobj_property(map_count,       CL_MEM_MAP_COUNT,       CL_uint)

#TODO: base...

#TODO: function get_info()
#TODO: function hostbuf()
#TODO: release()
#TODO: host_array()
#TODO: pointer

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)
