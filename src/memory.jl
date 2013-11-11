abstract CLMemObject

#This should be implemented by all subtypes
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
    if mem.id != C_NULL
        @check_release api.clReleaseMemObject(mem.id)
        mem.id = C_NULL
    end
    mem.valid = false
end

context(mem::CLMemObject) = begin
    param = Array(CL_context, 1)
    @check api.clGetMemObjectInfo(mem.id, CL_MEM_CONTEXT, 
                                  sizeof(Csize_t), param_value, C_NULL)
    return Context(param[1])
end


let mem_type(m::CLMemObject) = begin
        result = Array(CL_mem_object_type, 1)
        @check api.clGetMemObjectInfo(m.id, CL_MEM_TYPE, 
                        sizeof(CL_mem_object_type), result, C_NULL)
        return result[1]
    end

    size(m::CLMemObject) = begin
        result = Array(Csize_t, 1)
        @check api.clGetMemObjectInfo(m.id, CL_MEM_SIZE,
                        sizeof(Csize_t), result, C_NULL)
        return result[1]
    end

    reference_count(m::CLMemObject) = begin
        result = Array(CL_uint, 1)
        @check api.clGetMemObjectInfo(m.id, CL_MEM_REFERENCE_COUNT,
                        sizeof(CL_uint), result, C_NULL)
        return result[1]
    end

    map_count(m::CLMemObject) = begin
        result = Array(CL_uint, 1)
        @check api.clGetMemObjectInfo(m.id, CL_MEM_MAP_COUNT,
                        sizeof(CL_uint), result, C_NULL)
        return result[1]
    end

    info_map = (Symbol => Function)[
        :mem_type => mem_type,
        :size => size,
        :reference_count => reference_count,
        :map_count => map_count
    ]

    function info(mem::CLMemObject, minfo::Symbol)
        try
            func = info_map[minfo]
            func(mem)
        catch err
            if isa(err, KeyError)
                error("OpenCL.MemObject has no info for: $minfo")
            else
                throw(err)
            end
        end
    end
end


#TODO: function hostbuf()
#TODO: host_array()
#TODO: pointer

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)
