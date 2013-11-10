# low level OpenCL Device

immutable Device
    id :: CL_device_id
end

Base.pointer(d::Device) = d.id
@ocl_object_equality(Device) 

function Base.show(io::IO, d::Device)
    strip_extra_whitespace = r"\s+"
    device_name = replace(d[:name], strip_extra_whitespace, " ")
    platform_name = replace(d[:platform][:name], strip_extra_whitespace, " ")
    ptr_address = "0x$(hex(unsigned(Base.pointer(d)), WORD_SIZE>>2))"
    print(io, "<OpenCL.Device '$device_name' on '$platform_name' @$ptr_address>")
end

Base.getindex(d::Device, dinfo::Symbol) = info(d, dinfo)

#TODO: replace with int_info, str_info, etc...
macro device_property(func, cl_device_info, return_type)
    @eval begin
        function $func(d::Device)
            result = Array($return_type, 1)
            @check api.clGetDeviceInfo(d.id, $cl_device_info,
                                       sizeof($return_type), result, C_NULL)
            #TODO: see if there is a better way to do this 
            if $return_type  == CL_bool
                return bool(result[1])
            else
                return result[1]
            end
        end
    end
end

function info(d::Device, dinfo::CL_device_info)
    size = Array(Csize_t, 1)
    @check api.clGetDeviceInfo(d.id, dinfo, 0, C_NULL, size)
    result = Array(CL_char, size[1])
    @check api.clGetDeviceInfo(d.id, dinfo, size[1], result, C_NULL)
    return bytestring(convert(Ptr{CL_char}, result))
end

let driver_version(d::Device) = info(d, CL_DRIVER_VERSION)
    version(d::Device) = info(d, CL_DEVICE_VERSION)
    profile(d::Device) = info(d, CL_DEVICE_PROFILE)
    extensions(d::Device) = split(info(d, CL_DEVICE_EXTENSIONS))

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
        return bytestring(convert(Ptr{CL_char}, result))
    end

    @device_property(device_type, CL_DEVICE_TYPE,     CL_device_type)
   
    has_image_support(d::Device) = begin
        has_support = clbox(CL_FALSE)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_IMAGE_SUPPORT,
                                   sizeof(CL_bool), has_support, C_NULL)
        return bool(unbox(has_support) == CL_TRUE)
    end

    @device_property(queue_properties, CL_DEVICE_QUEUE_PROPERTIES, CL_command_queue_properties)

    has_queue_out_of_order_exec(d::Device) =
            bool(queue_properties(d) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)

    has_queue_profiling(d::Device) =
            bool(queue_properties(d) & CL_QUEUE_PROFILING_ENABLE)

    has_native_kernel(d::Device) = begin
        result = Array(CL_device_exec_capabilities, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_EXECUTION_CAPABILITIES,
                                   sizeof(CL_device_exec_capabilities), result, C_NULL)
        return bool(result[1] & CL_EXEC_NATIVE_KERNEL)
    end

    @device_property(vendor_id,             CL_DEVICE_VENDOR_ID,                CL_uint)
    @device_property(max_compute_units,     CL_DEVICE_MAX_COMPUTE_UNITS,        CL_uint)
    @device_property(max_work_item_dims,    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_uint)
    @device_property(max_clock_frequency,   CL_DEVICE_MAX_CLOCK_FREQUENCY,      CL_uint)
    @device_property(address_bits,          CL_DEVICE_ADDRESS_BITS,             CL_uint)
    @device_property(max_read_image_args,   CL_DEVICE_MAX_READ_IMAGE_ARGS,      CL_uint)
    @device_property(max_write_image_args,  CL_DEVICE_MAX_WRITE_IMAGE_ARGS,     CL_uint)
    @device_property(global_mem_size,       CL_DEVICE_GLOBAL_MEM_SIZE,          CL_ulong)
    @device_property(max_mem_alloc_size,    CL_DEVICE_MAX_MEM_ALLOC_SIZE,       CL_ulong)
    @device_property(max_const_buffer_size, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, CL_ulong)
    @device_property(local_mem_size,        CL_DEVICE_LOCAL_MEM_SIZE,           CL_ulong)

    has_local_mem(d::Device) = begin
        result = Array(CL_device_local_mem_type, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_LOCAL_MEM_TYPE,
                                   sizeof(CL_device_local_mem_type), result, C_NULL)
        return bool(result[1] == CL_LOCAL)
    end

    @device_property(host_unified_memory, CL_DEVICE_HOST_UNIFIED_MEMORY, CL_bool)
    @device_property(available,           CL_DEVICE_AVAILABLE,           CL_bool)
    @device_property(compiler_available,  CL_DEVICE_COMPILER_AVAILABLE,  CL_bool)

    max_work_item_sizes(d::Device) = begin
        dims = Array(CL_uint, 1)
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                   sizeof(CL_uint), dims, C_NULL)
        result = Array(Csize_t, dims[1])
        @check api.clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                   sizeof(Csize_t) * dims[1], result, C_NULL)
        return [r for r in result]
    end 

    @device_property(max_workgroup_size, CL_DEVICE_MAX_WORK_GROUP_SIZE, Csize_t)
    @device_property(max_parameter_size, CL_DEVICE_MAX_PARAMETER_SIZE,  Csize_t)
    @device_property(profiling_timer_resolution, CL_DEVICE_PROFILING_TIMER_RESOLUTION, Csize_t)

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

    #TODO: support haskeys, and keys by having
    # keys list on outsize of closure and generate 
    # map inside closure
    const info_map = (Symbol => Function)[
        :driver_version => driver_version,
        :version => profile,
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
        :max_work_item_sizes => max_work_item_sizes,
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
        :max_workgroup_size => max_workgroup_size, 
        :max_parameter_size => max_parameter_size,
        :profiling_timer_resolution => profiling_timer_resolution,
        :max_image2d_shape => max_image2d_shape,
        :max_image3d_shape => max_image3d_shape
    ]

    function info(d::Device, s::Symbol)
        try
            func = info_map[s]
            func(d)
        catch err
            if isa(err, KeyError)
                error("OpenCL.Device has no info for: $s")
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
        error("Unknown device type: $dtype")
    end
    return cl_dtype
end
