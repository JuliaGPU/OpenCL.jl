# low level OpenCL Device

immutable Device
    id :: CL_device_id
end

macro device_property(func, cl_device_info, return_type)
    quote
        function $(esc(func))(d::Device)
            result = Array($return_type, 1)
            clGetDeviceInfo(d.id, $cl_device_info, sizeof($return_type), result, C_NULL)
            #TODO: Find a way around this hack as CL_bool is typealiased to CL_uint
            if $(symbol(return_type)) == :CL_bool
                bool(result[1])
            else
                result[1]
            end
        end
    end
end


@ocl_func(clGetDeviceIDs, (CL_platform_id, CL_device_type, CL_uint, Ptr{CL_device_id}, Ptr{CL_uint}))
@ocl_func(clGetDeviceInfo, (CL_device_id, CL_device_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

function get_info(d::Device, info::CL_device_info)
    size = Array(Csize_t, 1)
    clGetDeviceInfo(d.id, info, 0, C_NULL, size)
    result = Array(CL_char, size[1])
    clGetDeviceInfo(d.id, info, size, result, C_NULL)
    return bytestring(convert(Ptr{CL_char}, result))
end

driver_version(d::Device) = get_info(d, CL_DRIVER_VERSION)
version(d::Device) = get_info(d, CL_DEVICE_VERSION)
profile(d::Device) = get_info(d, CL_DEVICE_PROFILE)
extensions(d::Device) = split(get_info(d, CL_DEVICE_EXTENSIONS))

@device_property(platform,    CL_DEVICE_PLATFORM, CL_platform_id)
@device_property(device_type, CL_DEVICE_TYPE,     CL_device_type)

box{T}(x::T) = T[x]
function has_image_support(d::Device)
    has_support = box(CL_FALSE)
    has_support[1] = CL_FALSE
    clGetDeviceInfo(d.id, CL_DEVICE_IMAGE_SUPPORT, sizeof(CL_bool), has_support, C_NULL)
    return has_support[1] == CL_TRUE ? true : false
end

function name(d::Device)
    size = Array(Csize_t, 1)
    clGetDeviceInfo(d.id, CL_DEVICE_NAME, 0, C_NULL, size)
    result = Array(CL_char, size[1])
    clGetDeviceInfo(d.id, CL_DEVICE_NAME, size[1] * sizeof(CL_char), result, C_NULL)
    return bytestring(convert(Ptr{CL_char}, result))
end

@device_property(queue_properties, CL_DEVICE_QUEUE_PROPERTIES, CL_command_queue_properties)

has_queue_out_of_order_exec(d::Device) = bool(queue_properties(d) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
has_queue_profiling(d::Device) = bool(queue_properties(d) & CL_QUEUE_PROFILING_ENABLE)

function has_native_kernel(d::Device)
    result = Array(CL_device_exec_capabilities, 1)
    clGetDeviceInfo(d.id, CL_DEVICE_EXECUTION_CAPABILITIES, sizeof(CL_device_exec_capabilities), result, C_NULL)
    return result[1] & CL_EXEC_NATIVE_KERNEL ? true : false
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

function has_local_mem(d::Device)
    result = Array(CL_device_local_mem_type, 1)
    clGetDeviceInfo(d.id, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(CL_device_local_mem_type), result, C_NULL)
    return result[1] == CL_LOCAL
end

@device_property(host_unified_memory, CL_DEVICE_HOST_UNIFIED_MEMORY, CL_bool)
@device_property(available,           CL_DEVICE_AVAILABLE,           CL_bool)
@device_property(compiler_available,  CL_DEVICE_COMPILER_AVAILABLE,  CL_bool)

# TODO: check in spec if these are size_t
function max_work_item_sizes(d::Device)
    dims = max_work_item_dims(d)
    result = Array(Csize_t, dims)
    clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(Csize_t) * dims, result, C_NULL)
    return [result[i] for i in 1:length(result)]
end 

@device_property(max_workgroup_size,         CL_DEVICE_MAX_WORK_GROUP_SIZE, Csize_t)
@device_property(max_parameter_size,         CL_DEVICE_MAX_PARAMETER_SIZE,  Csize_t)
@device_property(profiling_timer_resolution, CL_DEVICE_PROFILING_TIMER_RESOLUTION, Csize_t)

function max_image2d_shape(d::Device)
    width  = Array(Csize_t, 1)
    height = Array(Csize_t, 1)
    clGetDeviceInfo(d.id, CL_DEVICE_IMAGE2D_MAX_WIDTH,  sizeof(Csize_t), width,  C_NULL)
    clGetDeviceInfo(d.id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(Csize_t), height, C_NULL)
    return [width[1], height[1]]
end

function max_image3d_shape(d::Device)
    width  = Array(Csize_t, 1)
    height = Array(Csize_t, 1)
    depth =  Array(Csize_t, 1)
    clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_WIDTH,  sizeof(Csize_t), width,  C_NULL)
    clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(Csize_t), height, C_NULL)
    clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_DEPTH,  sizeof(Csize_t), depth,  C_NULL)
    return [width[1], height[1], depth[1]]
end
