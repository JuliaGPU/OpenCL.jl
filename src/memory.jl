abstract CLMemObject

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
                                  sizeof(Csize_t), param_value, C_NULL)
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

    info_map = (Symbol => Function)[
        :mem_type => mem_type,
        :mem_flags => mem_flags, 
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
                throw(ArgumentError("OpenCL.MemObject has no info for: $minfo"))
            else
                throw(err)
            end
        end
    end
end

function _symbols_to_cl_mem_flags(mem_flags::NTuple{2, Symbol})
    f_r  = :r  in mem_flags
    f_w  = :w  in mem_flags
    f_rw = :rw in mem_flags

    if f_r && f_w || f_r && f_rw || f_rw && f_w
        throw(ArgumentError("only one flag in {:r, :w, :rw} can be defined"))
    end

    flags::CL_mem_flags
    if f_rw && !(f_r || f_w)
        flags = CL_MEM_READ_WRITE
    elseif f_r && !(f_w || f_rw)
        flags = CL_MEM_READ_ONLY
    elseif f_w && !(f_r || f_rw)
        flags = CL_MEM_WRITE_ONLY
    else
        # default buffer is read/write
        flags = CL_MEM_READ_WRITE
    end
   
    f_alloc = :alloc in mem_flags
    f_use   = :use   in mem_flags
    f_copy  = :copy  in mem_flags
    if f_alloc && f_use || f_alloc && f_copy || f_use && f_copy
        throw(ArgumentError("only one flag in {:alloc, :use, :copy} can be defined"))
    end

    if f_alloc && !(f_use || f_copy)
        flags |= CL_MEM_ALLOC_HOST_PTR
    elseif f_use && !(f_alloc || f_copy) 
        flags |= CL_MEM_USE_HOST_PTR
    elseif f_copy && !(f_alloc || f_use)
        flags |= CL_MEM_COPY_HOST_PTR
    end
    return flags
end

#TODO: function hostbuf()
#TODO: host_array()
#TODO: pointer

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)
