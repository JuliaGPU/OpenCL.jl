# OpenCL Memory Object

abstract type CLMemObject <: CLObject end

#This should be implemented by all subtypes
# type CLMemType <: CLMemObject
#     valid::Bool
#     id::cl_mem
#     ...
# end

Base.unsafe_convert(::Type{cl_mem}, mem::CLMemObject) = mem.id

Base.pointer(mem::CLMemObject) = mem.id

Base.sizeof(mem::CLMemObject) = mem.size

function _finalize(mem::CLMemObject)
    if !mem.valid
        throw(CLMemoryError("attempted to double free mem object $mem"))
    end
    if mem.id != C_NULL
        clReleaseMemObject(mem.id)
        mem.id = C_NULL
    end
    mem.valid = false
end

context(mem::CLMemObject) = mem.context

function Base.getproperty(mem::CLMemObject, s::Symbol)
    if s == :context
        param = Ref{cl_context}()
        clGetMemObjectInfo(mem, CL_MEM_CONTEXT, sizeof(cl_context), param, C_NULL)
        return Context(param[], retain=true)
    elseif s == :mem_type
        result = Ref{cl_mem_object_type}()
        clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(cl_mem_object_type), result, C_NULL)
        return result[]
    elseif s == :mem_flags
        result = Ref{cl_mem_flags}()
        clGetMemObjectInfo(mem, CL_MEM_FLAGS, sizeof(cl_mem_flags), result, C_NULL)
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
    elseif s == :size
        result = Ref{Csize_t}()
        clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(Csize_t), result, C_NULL)
        return result[]
    elseif s == :reference_count
        result = Ref{Cuint}()
        clGetMemObjectInfo(mem, CL_MEM_REFERENCE_COUNT, sizeof(Cuint), result, C_NULL)
        return Int(result[])
    elseif s == :map_count
        result = Ref{Cuint}()
        clGetMemObjectInfo(mem, CL_MEM_MAP_COUNT, sizeof(Cuint), result, C_NULL)
        return Int(result[])
    else
        return getfield(mem, s)
    end
end

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)
