# OpenCL Memory Object

abstract type CLMemObject <: CLObject end

#This should be implemented by all subtypes
# type CLMemType <: CLMemObject
#     valid::Bool
#     id::CL_mem
#     ...
# end

Base.pointer(mem::CLMemObject) = mem.id

Base.sizeof(mem::CLMemObject) = begin
    val = Ref{Csize_t}(0)
    @check api.clGetMemObjectInfo(mem.id, CL_MEM_SIZE, sizeof(Csize_t),
                                  val, C_NULL)
    return val[]
end

function _finalize(mem::CLMemObject)
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
    param = Ref{CL_context}()
    @check api.clGetMemObjectInfo(mem.id, CL_MEM_CONTEXT,
                                  sizeof(Csize_t), param, C_NULL)
    return Context(param[], retain=true)
end

function info(mem::CLMemObject, minfo::Symbol)

    mem_type(m::CLMemObject) = begin
        result = Ref{CL_mem_object_type}()
        @check api.clGetMemObjectInfo(m.id, CL_MEM_TYPE,
                        sizeof(CL_mem_object_type), result, C_NULL)
        return result[]
    end

    mem_flags(m::CLMemObject) = begin
        result = Ref{CL_mem_flags}()
        @check api.clGetMemObjectInfo(m.id, CL_MEM_FLAGS,
                        sizeof(CL_mem_flags), result, C_NULL)
        mf = result[]
        flags = Symbol[]
        if (mf & CL_MEM_READ_WRITE) != 0
            push!(flags, :rw)
        end
        if (mf & CL_MEM_WRITE_ONLY) != 0
            push!(flags, :w)
        end
        if (mf & CL_MEM_READ_ONLY) != 0
            push!(flags, :r)
        end
        if (mf & CL_MEM_USE_HOST_PTR) != 0
            push!(flags, :use)
        end
        if (mf & CL_MEM_ALLOC_HOST_PTR) != 0
            push!(flags, :alloc)
        end
        if (mf & CL_MEM_COPY_HOST_PTR) != 0
            push!(flags, :copy)
        end
        return tuple(flags...)
    end

    size(m::CLMemObject) = begin
        result = Ref{Csize_t}()
        @check api.clGetMemObjectInfo(m.id, CL_MEM_SIZE,
                        sizeof(Csize_t), result, C_NULL)
        return result[]
    end

    reference_count(m::CLMemObject) = begin
        result = Ref{CL_uint}()
        @check api.clGetMemObjectInfo(m.id, CL_MEM_REFERENCE_COUNT,
                        sizeof(CL_uint), result, C_NULL)
        return result[]
    end

    map_count(m::CLMemObject) = begin
        result = Ref{CL_uint}()
        @check api.clGetMemObjectInfo(m.id, CL_MEM_MAP_COUNT,
                        sizeof(CL_uint), result, C_NULL)
        return result[]
    end

    info_map = Dict{Symbol, Function}(
        :mem_type => mem_type,
        :mem_flags => mem_flags,
        :size => size,
        :reference_count => reference_count,
        :map_count => map_count
    )

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

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)
