# OpenCL Memory Object

abstract CLMemObject <: CLObject

#This should be implemented by all subtypes
# type CLMemType <: CLMemObject
#     valid::Bool
#     id::CL_mem
#     ...
# end

Base.pointer(mem::CLMemObject) = mem.id

Base.sizeof(mem::CL_mem) = begin
    val = Csize_t[0,]
    @check api.clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(Csize_t), 
                                  val, C_NULL)
    return val[1]
end

function release!(mem::CLMemObject)
    if !mem.valid
        throw(CLMemoryError("attempted to double free mem object $mem"))
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
                                  sizeof(Csize_t), param, C_NULL)
    return Context(param[1], retain=true)
end


let mem_type(m::CLMemObject) = begin
        result = Array(CL_mem_object_type, 1)
        @check api.clGetMemObjectInfo(m.id, CL_MEM_TYPE, 
                        sizeof(CL_mem_object_type), result, C_NULL)
        return result[1]
    end

    mem_flags(m::CLMemObject) = begin
        result = Array(CL_mem_flags)
        @check api.clGetMemObjectInfo(m.id, CL_MEM_FLAGS,
                        sizeof(CL_mem_flags), result, C_NULL)
        mf = result[1]
        flags = Symbol[]
        if bool(mf & CL_MEM_READ_WRITE)
            push!(flags, :rw)
        end
        if bool(mf & CL_MEM_WRITE_ONLY)
            push!(flags, :w)
        end
        if bool(mf & CL_MEM_READ_ONLY)
            push!(flags, :r)
        end
        if bool(mf & CL_MEM_USE_HOST_PTR)
            push!(flags, :use)
        end
        if bool(mf & CL_MEM_ALLOC_HOST_PTR)
            push!(flags, :alloc)
        end
        if bool(mf & CL_MEM_COPY_HOST_PTR)
            push!(flags, :copy)
        end
        return tuple(flags...)
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

    info_map = @compat Dict{Symbol, Function}(
        :mem_type => mem_type,
        :mem_flags => mem_flags, 
        :size => size,
        :reference_count => reference_count,
        :map_count => map_count
    )

    function info(mem::CLMemObject, minfo::Symbol)
        try
            func = info_map[minfo]
            func(mem)
        catch err
            if isa(err, KeyError)
                throw(ArgumentError("OpenCL.MemObject has no info for: $minfo"))
            else
                throw(err)
            end
        end
    end
end

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)
