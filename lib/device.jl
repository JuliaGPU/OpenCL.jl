# OpenCL.Device

struct Device <: CLObject
    id::cl_device_id
end

Base.unsafe_convert(::Type{cl_device_id}, d::Device) = d.id

Base.pointer(d::Device) = d.id

function Base.show(io::IO, d::Device)
    strip_extra_whitespace = r"\s+"
    device_name = replace(d.name, strip_extra_whitespace => " ")
    platform_name = replace(d.platform.name, strip_extra_whitespace => " ")
    ptr_val = convert(UInt, pointer(d))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Device($device_name on $platform_name @$ptr_address)")
end

function Base.getproperty(d::Device, s::Symbol)
    # simple string properties
    string_properties = Dict(
        :profile        => CL_DEVICE_PROFILE,
        :version        => CL_DEVICE_VERSION,
        :driver_version => CL_DRIVER_VERSION,
        :name           => CL_DEVICE_NAME,
    )
    if haskey(string_properties, s)
        size = Ref{Csize_t}()
        clGetDeviceInfo(d, string_properties[s], 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d, string_properties[s], size[], result, C_NULL)
        return CLString(result)
    end

    # scalar values
    int_properties = Dict(
        :queue_properties           => (CL_DEVICE_QUEUE_PROPERTIES, cl_command_queue_properties),
        :exec_capabilities          => (CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities),
        :vendor_id                  => (CL_DEVICE_VENDOR_ID,                 Cuint),
        :max_compute_units          => (CL_DEVICE_MAX_COMPUTE_UNITS,         Cuint),
        :max_work_item_dims         => (CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,  Cuint),
        :max_clock_frequency        => (CL_DEVICE_MAX_CLOCK_FREQUENCY,       Cuint),
        :address_bits               => (CL_DEVICE_ADDRESS_BITS,              Cuint),
        :max_read_image_args        => (CL_DEVICE_MAX_READ_IMAGE_ARGS,       Cuint),
        :max_write_image_args       => (CL_DEVICE_MAX_WRITE_IMAGE_ARGS,      Cuint),
        :global_mem_size            => (CL_DEVICE_GLOBAL_MEM_SIZE,           Culong),
        :max_mem_alloc_size         => (CL_DEVICE_MAX_MEM_ALLOC_SIZE,        Culong),
        :max_const_buffer_size      => (CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,  Culong),
        :local_mem_size             => (CL_DEVICE_LOCAL_MEM_SIZE,            Culong),
        :max_work_group_size        => (CL_DEVICE_MAX_WORK_GROUP_SIZE,       Csize_t),
        :max_parameter_size         => (CL_DEVICE_MAX_PARAMETER_SIZE,        Csize_t),
        :profiling_timer_resolution => (CL_DEVICE_PROFILING_TIMER_RESOLUTION, Csize_t),
    )
    if haskey(int_properties, s)
        prop, typ = int_properties[s]
        result = Ref{typ}()
        clGetDeviceInfo(d, prop, sizeof(typ), result, C_NULL)
        return result[]
    end

    # boolean properties
    bool_properties = Dict(
        :has_image_support           => CL_DEVICE_IMAGE_SUPPORT,
        :has_local_mem               => CL_DEVICE_LOCAL_MEM_TYPE,
        :host_unified_memory         => CL_DEVICE_HOST_UNIFIED_MEMORY,
        :available                   => CL_DEVICE_AVAILABLE,
        :compiler_available          => CL_DEVICE_COMPILER_AVAILABLE
    )
    if haskey(bool_properties, s)
        result = Ref{cl_bool}()
        clGetDeviceInfo(d, bool_properties[s], sizeof(cl_bool), result, C_NULL)
        return result[] == CL_TRUE
    end

    # boolean queue properties
    # TODO: move this to `queue_info`?
    queue_properties = Dict(
        :has_queue_out_of_order_exec => CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        :has_queue_profiling         => CL_QUEUE_PROFILING_ENABLE,
    )
    if haskey(queue_properties, s)
        return d.queue_properties & queue_properties[s] != 0
    end

    # boolean execution properties
    # TODO: move this to `execution_info`?
    exec_properties = Dict(
        :has_native_kernel => CL_EXEC_NATIVE_KERNEL,
    )
    if haskey(exec_properties, s)
        return d.exec_capabilities & exec_properties[s] != 0
    end

    if s == :extensions
        size = Ref{Csize_t}()
        clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, size[], result, C_NULL)
        bs = CLString(result)
        return String[string(s) for s in split(bs)]
    end

    if s == :platform
        result = Ref{cl_platform_id}()
        clGetDeviceInfo(d, CL_DEVICE_PLATFORM,
                        sizeof(cl_platform_id), result, C_NULL)
        return Platform(result[])
    end

    if s == :device_type
        result = Ref{cl_device_type}()
        clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(cl_device_type), result, C_NULL)
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

    if s == :max_work_item_size
        dims = Ref{Cuint}()
        clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(Cuint), dims, C_NULL)
        result = Vector{Csize_t}(undef, dims[])
        clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(Csize_t) * dims[], result, C_NULL)
        return tuple([Int(r) for r in result]...)
    end

    if s == :max_image2d_shape
        width  = Ref{Csize_t}()
        height = Ref{Csize_t}()
        clGetDeviceInfo(d, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(Csize_t), width,  C_NULL)
        clGetDeviceInfo(d, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(Csize_t), height, C_NULL)
        return (width[], height[])
    end

    if s == :max_image3d_shape
        width  = Ref{Csize_t}()
        height = Ref{Csize_t}()
        depth =  Ref{Csize_t}()
        clGetDeviceInfo(d, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(Csize_t), width, C_NULL)
        clGetDeviceInfo(d, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(Csize_t), height, C_NULL)
        clGetDeviceInfo(d, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(Csize_t), depth, C_NULL)
        return (width[], height[], depth[])
    end

    return getfield(d, s)
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
