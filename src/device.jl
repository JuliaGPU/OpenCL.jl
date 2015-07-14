# OpenCL.Device

immutable Device <: CLObject
    id :: CL_device_id
end

Base.pointer(d::Device) = d.id

function Base.show(io::IO, d::Device)
    strip_extra_whitespace = r"\s+"
    device_name = replace(d[:name], strip_extra_whitespace, " ")
    platform_name = replace(d[:platform][:name], strip_extra_whitespace, " ")
    ptr_address = "0x$(hex(unsigned(Base.pointer(d)), WORD_SIZE>>2))"
    print(io, "OpenCL.Device($device_name on $platform_name @$ptr_address)")
end

Base.getindex(d::Device, dinfo::Symbol) = info(d, dinfo)

macro int_info(func, cl_device_info, return_type)
    quote
        function $(esc(func))(d::Device)
            result = Array($return_type, 1)
            @check api.clGetDeviceInfo(d.id, $cl_device_info,
                                       sizeof($return_type), result, C_NULL)
            return result[1]
        end
    end
end

let profile(d::Device) = begin
        size = Array(Csize_t, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_PROFILE, 0, C_NULL, size)
        result = Array(CL_char, size[1])
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_PROFILE, size[1], result, C_NULL)
        bs = bytestring(convert(Ptr{CL_char}, result))
        return bs
    end

    version(d::Device) = begin
        size = Array(Csize_t, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_VERSION, 0, C_NULL, size)
        result = Array(CL_char, size[1])
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_VERSION, size[1], result, C_NULL)
        bs = bytestring(convert(Ptr{CL_char}, result))
        return bs
    end

    driver_version(d::Device) = begin
        size = Array(Csize_t, 1)
        @check api.clGetDeviceInfo(d.id, CL_DRIVER_VERSION, 0, C_NULL, size)
        result = Array(CL_char, size[1])
        @check api.clGetDeviceInfo(d.id, CL_DRIVER_VERSION, size[1], result, C_NULL)
        bs = bytestring(convert(Ptr{CL_char}, result))
        return string(replace(bs, r"\s+", " "))
    end

    extensions(d::Device) = begin
        size = Array(Csize_t, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_EXTENSIONS, 0, C_NULL, size)
        result = Array(CL_char, size[1])
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_EXTENSIONS, size[1], result, C_NULL)
        bs = bytestring(convert(Ptr{CL_char}, result))
        return String[string(s) for s in split(bs)]
    end

    platform(d::Device) = begin
        result = Array(CL_platform_id, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_PLATFORM,
                                   sizeof(CL_platform_id), result, C_NULL)
        return Platform(result[1])
    end

    name(d::Device) = begin
        size = Array(Csize_t, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_NAME, 0, C_NULL, size)
        result = Array(CL_char, size[1])
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_NAME,
                                   size[1] * sizeof(CL_char), result, C_NULL)
        n = bytestring(convert(Ptr{Cchar}, result))
        return string(replace(n, r"\s+", " "))
    end

    @int_info(device_type, CL_DEVICE_TYPE, CL_device_type)
    device_type(d::Device) = begin
        result = Array(CL_device_type, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_TYPE,
                                   sizeof(CL_device_type), result, C_NULL)
        result = result[1]
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

    has_image_support(d::Device) = begin
        has_support = CL_bool[CL_FALSE]
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE_SUPPORT,
                                   sizeof(CL_bool), has_support, C_NULL)
        return has_support[1] == CL_TRUE
    end

    @int_info(queue_properties, CL_DEVICE_QUEUE_PROPERTIES, CL_command_queue_properties)

    has_queue_out_of_order_exec(d::Device) =
        (queue_properties(d) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0

    has_queue_profiling(d::Device) =
        (queue_properties(d) & CL_QUEUE_PROFILING_ENABLE) != 0

    has_native_kernel(d::Device) = begin
        result = Array(CL_device_exec_capabilities, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_EXECUTION_CAPABILITIES,
                                   sizeof(CL_device_exec_capabilities), result, C_NULL)
        return (result[1] & CL_EXEC_NATIVE_KERNEL) != 0
    end

    @int_info(vendor_id,             CL_DEVICE_VENDOR_ID,                CL_uint)
    @int_info(max_compute_units,     CL_DEVICE_MAX_COMPUTE_UNITS,        CL_uint)
    @int_info(max_work_item_dims,    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_uint)
    @int_info(max_clock_frequency,   CL_DEVICE_MAX_CLOCK_FREQUENCY,      CL_uint)
    @int_info(address_bits,          CL_DEVICE_ADDRESS_BITS,             CL_uint)
    @int_info(max_read_image_args,   CL_DEVICE_MAX_READ_IMAGE_ARGS,      CL_uint)
    @int_info(max_write_image_args,  CL_DEVICE_MAX_WRITE_IMAGE_ARGS,     CL_uint)
    @int_info(global_mem_size,       CL_DEVICE_GLOBAL_MEM_SIZE,          CL_ulong)
    @int_info(max_mem_alloc_size,    CL_DEVICE_MAX_MEM_ALLOC_SIZE,       CL_ulong)
    @int_info(max_const_buffer_size, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, CL_ulong)
    @int_info(local_mem_size,        CL_DEVICE_LOCAL_MEM_SIZE,           CL_ulong)

    has_local_mem(d::Device) = begin
        result = Array(CL_device_local_mem_type, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_LOCAL_MEM_TYPE,
                                   sizeof(CL_device_local_mem_type), result, C_NULL)
        return result[1] == CL_LOCAL
    end

    host_unified_memory(d::Device) = begin
        result = Array(CL_bool, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                                   sizeof(CL_bool), result, C_NULL)
        return result[1] != 0
    end

    available(d::Device) = begin
        result = Array(CL_bool, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_AVAILABLE,
                                   sizeof(CL_bool), result, C_NULL)
        return result[1] != 0
    end

    compiler_available(d::Device) = begin
        result = Array(CL_bool, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_COMPILER_AVAILABLE,
                                   sizeof(CL_bool), result, C_NULL)
        return result[1] != 0
    end

    max_work_item_size(d::Device) = begin
        dims = Array(CL_uint, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                   sizeof(CL_uint), dims, C_NULL)
        result = Array(Csize_t, dims[1])
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                   sizeof(Csize_t) * dims[1], result, C_NULL)
        return @compat tuple([Int(r) for r in result]...)
    end

    @int_info(max_work_group_size, CL_DEVICE_MAX_WORK_GROUP_SIZE, Csize_t)
    @int_info(max_parameter_size, CL_DEVICE_MAX_PARAMETER_SIZE,  Csize_t)
    @int_info(profiling_timer_resolution, CL_DEVICE_PROFILING_TIMER_RESOLUTION, Csize_t)

    max_image2d_shape(d::Device) = begin
        width  = Array(Csize_t, 1)
        height = Array(Csize_t, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                                   sizeof(Csize_t), width,  C_NULL)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                                   sizeof(Csize_t), height, C_NULL)
        return (width[1], height[1])
    end

    max_image3d_shape(d::Device) = begin
        width  = Array(Csize_t, 1)
        height = Array(Csize_t, 1)
        depth =  Array(Csize_t, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_WIDTH,
                                   sizeof(Csize_t), width, C_NULL)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                                   sizeof(Csize_t), height, C_NULL)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_DEPTH,
                                   sizeof(Csize_t), depth, C_NULL)
        return (width[1], height[1], depth[1])
    end

    const info_map = @compat Dict{Symbol, Function}(
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

    function info(d::Device, s::Symbol)
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
