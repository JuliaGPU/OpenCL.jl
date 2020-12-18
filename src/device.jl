# OpenCL.Device

struct Device <: CLObject
    id :: CL_device_id
end

Base.pointer(d::Device) = d.id

function Base.show(io::IO, d::Device)
    strip_extra_whitespace = r"\s+"
    device_name = replace(d[:name], strip_extra_whitespace => " ")
    platform_name = replace(d[:platform][:name], strip_extra_whitespace => " ")
    ptr_val = convert(UInt, Base.pointer(d))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Device($device_name on $platform_name @$ptr_address)")
end

Base.getindex(d::Device, dinfo::Symbol) = info(d, dinfo)

macro int_info(func, cl_device_info, return_type)
    quote
        function $(esc(func))(d::Device)
            result = Ref{$return_type}()
            @check api.clGetDeviceInfo(d.id, $cl_device_info,
                                       sizeof($return_type), result, C_NULL)
            return result[]
        end
    end
end

"""
    profile(device)::String

Gets the device's profile (`CL_DEVICE_PROFILE`). Will be one of `"FULL_PROFILE"`, `"EMBEDDED_PROFILE"`.
"""
profile(d::Device) = begin
    size = Ref{Csize_t}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_PROFILE, 0, C_NULL, size)
    result = Vector{CL_char}(undef, size[])
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_PROFILE, size[], result, C_NULL)
    bs = CLString(result)
    return bs
end

"""
    version(device)::String

Gets the OpenCL version string (`CL_DEVICE_VERSION`).
"""
version(d::Device) = begin
    size = Ref{Csize_t}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_VERSION, 0, C_NULL, size)
    result = Vector{CL_char}(undef, size[])
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_VERSION, size[], result, C_NULL)
    bs = CLString(result)
    return bs
end

"""
    driver_version(device)::String

Gets the OpenCL driver version for the device (`CL_DRIVER_VERSION`). May contain a build date or other information.
"""
driver_version(d::Device) = begin
    size = Ref{Csize_t}()
    @check api.clGetDeviceInfo(d.id, CL_DRIVER_VERSION, 0, C_NULL, size)
    result = Vector{CL_char}(undef, size[])
    @check api.clGetDeviceInfo(d.id, CL_DRIVER_VERSION, size[], result, C_NULL)
    bs = CLString(result)
    return string(replace(bs, r"\s+" => " "))
end

"""
    extensions(device)::Vector{String}

Gets OpenCL extensions for the device (`CL_DEVICE_EXTENSIONS`).
"""
extensions(d::Device) = begin
    size = Ref{Csize_t}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_EXTENSIONS, 0, C_NULL, size)
    result = Vector{CL_char}(undef, size[])
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_EXTENSIONS, size[], result, C_NULL)
    bs = CLString(result)
    return String[string(s) for s in split(bs)]
end

"""
    platform(device)::Platform

Gets the `cl.Platform` that the device is associated with (`CL_DEVICE_PLATFORM`).
"""
platform(d::Device) = begin
    result = Ref{CL_platform_id}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_PLATFORM,
                                sizeof(CL_platform_id), result, C_NULL)
    return Platform(result[])
end

"""
    name(device)::String

Gets the name string for the device (`CL_DEVICE_NAME`).
"""
name(d::Device) = begin
    size = Ref{Csize_t}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_NAME, 0, C_NULL, size)
    result = Vector{CL_char}(undef, size[])
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_NAME,
                                size[] * sizeof(CL_char), result, C_NULL)
    n = CLString(result)
    return string(replace(n, r"\s+" => " "))
end

"""
    device_type(device)::Symbol

Gets the device type (`CL_DEVICE_TYPE`). Will be one of `:gpu`, `:cpu`, `:accelerator`, `:custom`, or `:unknown`.
"""
device_type(d::Device) = begin
    result = Ref{CL_device_type}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_TYPE,
                                sizeof(CL_device_type), result, C_NULL)
    result = result[]
    if result == CL_DEVICE_TYPE_GPU
        return :gpu
    elseif result == CL_DEVICE_TYPE_CPU
        return :cpu
    elseif result == CL_DEVICE_TYPE_ACCELERATOR
        return :accelerator
    elseif result == CL_DEVICE_TYPE_CUSTOM
        return :custom
    else
        return :unknown
    end
end

"""
    has_image_support(device)::Bool

Does the device have image support (`CL_DEVICE_IMAGE_SUPPORT`)?
"""
has_image_support(d::Device) = begin
    has_support = Ref{CL_bool}(CL_FALSE)
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE_SUPPORT,
                                sizeof(CL_bool), has_support, C_NULL)
    return has_support[] == CL_TRUE
end

"""
    queue_properties(device)::UInt

Returns the command-queue properties supported, as a bit field (`CL_DEVICE_QUEUE_PROPERTIES`).

!!! compat "Deprecated"
    This method is deprecated as of version 2.0.
"""
@int_info(queue_properties, CL_DEVICE_QUEUE_PROPERTIES, CL_command_queue_properties)

"""
    has_queue_out_of_order_exec(device)::Bool

Does the device have out of order execution (`CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`)?
"""
has_queue_out_of_order_exec(d::Device) =
    (queue_properties(d) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0

"""
    has_queue_profiling(device)::Bool

Does the device have queue profiling (`CL_QUEUE_PROFILING_ENABLE`)?
"""
has_queue_profiling(d::Device) =
    (queue_properties(d) & CL_QUEUE_PROFILING_ENABLE) != 0

"""
    has_native_kernel(device)::Bool

Does the device have native kernels (`CL_EXEC_NATIVE_KERNEL`)?
"""
has_native_kernel(d::Device) = begin
    result = Ref{CL_device_exec_capabilities}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_EXECUTION_CAPABILITIES,
                                sizeof(CL_device_exec_capabilities), result, C_NULL)
    return (result[] & CL_EXEC_NATIVE_KERNEL) != 0
end

"""
    vendor_id(device)::UInt

Gets the unique vendor identifier (`CL_DEVICE_VENDOR_ID`).
"""
@int_info(vendor_id,             CL_DEVICE_VENDOR_ID,                CL_uint)

"""
    max_compute_units(device)::UInt

Gets the maximum number of compute units for the device (`CL_DEVICE_MAX_COMPUTE_UNITS`).
"""
@int_info(max_compute_units,     CL_DEVICE_MAX_COMPUTE_UNITS,        CL_uint)

"""
    max_work_item_dims(device)::UInt

Gets the maximum work item dimensions for the device (`CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS`). Almost certainly three.
"""
@int_info(max_work_item_dims,    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_uint)

"""
    max_clock_frequency(device)::UInt

Maximum clock frequency of the the device [MHz] (`CL_DEVICE_MAX_CLOCK_FREQUENCY`).
"""
@int_info(max_clock_frequency,   CL_DEVICE_MAX_CLOCK_FREQUENCY,      CL_uint)

"""
    address_bits(device)::UInt

The address size of the device (`CL_DEVICE_ADDRESS_BITS`). Either 32 or 64.
"""
@int_info(address_bits,          CL_DEVICE_ADDRESS_BITS,             CL_uint)

"""
    max_read_image_args(device)::UInt

Maximum number of simultaenous image objects that can be read (`CL_DEVICE_MAX_READ_IMAGE_ARGS`).
"""
@int_info(max_read_image_args,   CL_DEVICE_MAX_READ_IMAGE_ARGS,      CL_uint)

"""
    max_write_image_args(device)::UInt

Maximum number of simultaenous image objects that can be written to (`CL_DEVICE_MAX_WRITE_IMAGE_ARGS`).
"""
@int_info(max_write_image_args,  CL_DEVICE_MAX_WRITE_IMAGE_ARGS,     CL_uint)

"""
    global_mem_size(device)::UInt

Bytes of global memory on the device (`CL_DEVICE_GLOBAL_MEM_SIZE`).
"""
@int_info(global_mem_size,       CL_DEVICE_GLOBAL_MEM_SIZE,          CL_ulong)

"""
    max_mem_alloc_size(device)::UInt

Maximum size (in bytes) that can be allocated in a single chunk (`CL_DEVICE_MAX_MEM_ALLOC_SIZE`). On most devices, will be the larger of 1/4th the global memory size (`global_mem_size(device)`) and `128 ⋅ 1024 ⋅ 1024` (128 MiB).
"""
@int_info(max_mem_alloc_size,    CL_DEVICE_MAX_MEM_ALLOC_SIZE,       CL_ulong)

"""
    max_const_buffer_size(device)::UInt

Maximum size (in bytes) of a buffer declared constant (`CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE`).
"""
@int_info(max_const_buffer_size, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, CL_ulong)

"""
    local_mem_size(device)::UInt

Bytes of local memory on the device (`CL_DEVICE_LOCAL_MEM_SIZE`).
"""
@int_info(local_mem_size,        CL_DEVICE_LOCAL_MEM_SIZE,           CL_ulong)

"""
    has_local_mem(device)::Bool

Does the device have local memory (`CL_DEVICE_LOCAL_MEM_TYPE`)?
"""
has_local_mem(d::Device) = begin
    result = Ref{CL_device_local_mem_type}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_LOCAL_MEM_TYPE,
                                sizeof(CL_device_local_mem_type), result, C_NULL)
    return result[] == CL_LOCAL
end

"""
    host_unified_memory(device)::Bool

Do the device and host have a unified memory system (`CL_DEVICE_HOST_UNIFIED_MEMORY`)?

!!! compat "Deprecated"
    This function is missing before version 1.1 and is deprecated as of version 2.0.
"""
host_unified_memory(d::Device) = begin
    result = Ref{CL_bool}(CL_FALSE)
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                                sizeof(CL_bool), result, C_NULL)
    return result[] == CL_TRUE
end

"""
    available(device)::Bool

Is the device available (`CL_DEVICE_AVAILABLE`)?
"""
available(d::Device) = begin
    result = Ref{CL_bool}(CL_FALSE)
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_AVAILABLE,
                                sizeof(CL_bool), result, C_NULL)
    return result[] == CL_TRUE
end

"""
    compiler_available(device)::Bool

Is there a compiler available for the device (`CL_DEVICE_COMPILER_AVAILABLE`)? This can only be false for embedded devices.
"""
compiler_available(d::Device) = begin
    result = Ref{CL_bool}(CL_FALSE)
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_COMPILER_AVAILABLE,
                                sizeof(CL_bool), result, C_NULL)
    return result[] == CL_TRUE
end

"""
    max_work_item_size(device)::Union{Tuple{Int}, Tuple{Int, Int}, Tuple{Int, Int, Int}}

Maximum size, in each dimension, of work item IDs (`CL_DEVICE_MAX_WORK_ITEM_SIZES`).

!!! note "Note"
    This is _not_ the largest size of work items that can be launched. The product of the size in each dimension must not be greater than `cl.max_work_group_size(device)`. This is just the upper bound in each dimension.
"""
max_work_item_size(d::Device) = begin
    dims = Ref{CL_uint}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                sizeof(CL_uint), dims, C_NULL)
    result = Vector{Csize_t}(undef, dims[])
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                sizeof(Csize_t) * dims[], result, C_NULL)
    return tuple([Int(r) for r in result]...)
end

"""
    max_work_group_size(device)::UInt

The largest number of work items that the device can run in a single work group (`CL_DEVICE_MAX_WORK_GROUP_SIZE`).
"""
@int_info(max_work_group_size, CL_DEVICE_MAX_WORK_GROUP_SIZE, Csize_t)

"""
    max_parameter_size(device)::UInt

Maximum size (in bytes) of a parameter to a kernel (`CL_DEVICE_MAX_PARAMETER_SIZE`).
"""
@int_info(max_parameter_size, CL_DEVICE_MAX_PARAMETER_SIZE,  Csize_t)

"""
    profiling_timer_resolution(device)::UInt

The resolution of the device's profiling timer in ns (`CL_DEVICE_PROFILING_TIMER_RESOLUTION`).
"""
@int_info(profiling_timer_resolution, CL_DEVICE_PROFILING_TIMER_RESOLUTION, Csize_t)

"""
    max_image2d_shape(device)::Tuple{UInt, UInt}

The maximum dimensions of a 2D image on the device (`CL_DEVICE_IMAGE2D_MAX_WIDTH`, `CL_DEVICE_IMAGE2D_MAX_HEIGHT`)/
"""
max_image2d_shape(d::Device) = begin
    width  = Ref{Csize_t}()
    height = Ref{Csize_t}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                                sizeof(Csize_t), width,  C_NULL)
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                                sizeof(Csize_t), height, C_NULL)
    return (width[], height[])
end

"""
    max_image3d_shape(device)::Tuple{UInt, UInt, UInt}

The maximum dimensions of a 3D image on the device (`CL_DEVICE_IMAGE3D_MAX_WIDTH`, `CL_DEVICE_IMAGE3D_MAX_HEIGHT`, `CL_DEVICE_IMAGE3D_MAX_DEPTH`).
"""
max_image3d_shape(d::Device) = begin
    width  = Ref{Csize_t}()
    height = Ref{Csize_t}()
    depth =  Ref{Csize_t}()
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_WIDTH,
                                sizeof(Csize_t), width, C_NULL)
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                                sizeof(Csize_t), height, C_NULL)
    @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_DEPTH,
                                sizeof(Csize_t), depth, C_NULL)
    return (width[], height[], depth[])
end

function info(d::Device, s::Symbol)
    info_map = Dict{Symbol, Function}(
        :driver_version => driver_version,
        :version => version,
        :profile => profile,
        :extensions => extensions,
        :platform => platform,
        :name => name,
        :device_type => device_type,
        :has_image_support => has_image_support,
        :queue_properties => queue_properties,
        :has_queue_out_of_order_exec => has_queue_out_of_order_exec,
        :has_queue_profiling => has_queue_profiling,
        :has_native_kernel => has_native_kernel,
        :vendor_id => vendor_id,
        :max_compute_units => max_compute_units,
        :max_work_item_size => max_work_item_size,
        :max_clock_frequency => max_clock_frequency,
        :address_bits => address_bits,
        :max_read_image_args => max_read_image_args,
        :max_write_image_args => max_write_image_args,
        :global_mem_size => global_mem_size,
        :max_mem_alloc_size => max_mem_alloc_size,
        :max_const_buffer_size => max_const_buffer_size,
        :local_mem_size => local_mem_size,
        :has_local_mem => has_local_mem,
        :host_unified_memory => host_unified_memory,
        :available => available,
        :compiler_available => compiler_available,
        :max_work_group_size => max_work_group_size,
        :max_work_item_dims => max_work_item_dims,
        :max_parameter_size => max_parameter_size,
        :profiling_timer_resolution => profiling_timer_resolution,
        :max_image2d_shape => max_image2d_shape,
        :max_image3d_shape => max_image3d_shape
    )

    try
        func = info_map[s]
        func(d)
    catch err
        if isa(err, KeyError)
            throw(ArgumentError("OpenCL.Device has no info for: $s"))
        else
            throw(err)
        end
    end
end

function cl_device_type(dtype::Symbol)
    if dtype == :all
        cl_dtype = CL_DEVICE_TYPE_ALL
    elseif dtype == :cpu
        cl_dtype = CL_DEVICE_TYPE_CPU
    elseif dtype == :gpu
        cl_dtype = CL_DEVICE_TYPE_GPU
    elseif dtype == :accelerator
        cl_dtype = CL_DEVICE_TYPE_ACCELERATOR
    elseif dtype == :custom
        cl_dtype = CL_DEVICE_TYPE_CUSTOM
    elseif dtype == :default
        cl_dtype = CL_DEVICE_TYPE_DEFAULT
    else
        throw(ArgumentError("Unknown device type: $dtype"))
    end
    return cl_dtype
end
