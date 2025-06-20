# Raw memory management

abstract type AbstractMemoryObject <: CLObject end
abstract type AbstractPointerMemory end
const AbstractMemory = Union{AbstractMemoryObject, AbstractPointerMemory}

# this will be specialized for each memory type
Base.convert(T::Type{<:Union{Ptr, CLPtr}}, mem::AbstractMemory) =
    throw(ArgumentError("Illegal conversion of a $(typeof(mem)) to a $T"))

# ccall integration
#
# taking the pointer of a memory object means returning the underlying pointer,
# and not the pointer of the object itself.
Base.unsafe_convert(P::Type{<:Union{Ptr, CLPtr}}, mem::AbstractMemory) = convert(P, mem)


## opaque memory objects

# This should be implemented by all subtypes
#type MemoryType <: AbstractMemoryObject
#    id::cl_mem
#    ...
#end

Base.sizeof(mem::AbstractMemoryObject) = mem.size

release(mem::AbstractMemoryObject) = clReleaseMemObject(mem)

function Base.getproperty(mem::AbstractMemoryObject, s::Symbol)
    if s == :type
        result = Ref{cl_mem_object_type}()
        clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(cl_mem_object_type), result, C_NULL)
        return result[]
    elseif s == :flags
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
    elseif s == :device_address
        result = Ref{cl_mem_device_address_ext}()
        clGetMemObjectInfo(mem, CL_MEM_DEVICE_ADDRESS_EXT, sizeof(cl_mem_device_address_ext), result, C_NULL)
        return CLPtr{Cvoid}(result[])
    else
        return getfield(mem, s)
    end
end

# for passing buffers to OpenCL APIs: use the underlying handle
Base.unsafe_convert(::Type{cl_mem}, mem::AbstractMemoryObject) = mem.id

# for passing buffers to kernels: pass the private device pointer
Base.convert(::Type{CLPtr{T}}, mem::AbstractMemoryObject) where {T} =
    convert(CLPtr{T}, pointer(mem))

include("memory/buffer.jl")

#TODO: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)
#TODO: enqueue_migrate_mem_objects_ext(queue, mem_objects, flags=0, wait_for=None)


## pointer-based memory

include("memory/usm.jl")
include("memory/svm.jl")
