# OpenCL.Device

struct Device <: CLObject
    id::cl_device_id
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
            clGetDeviceInfo(d.id, $cl_device_info,
                                       sizeof($return_type), result, C_NULL)
            return result[]
        end
    end
end

function info(d::Device, s::Symbol)

    profile(d::Device) = begin
        size = Ref{Csize_t}()
        clGetDeviceInfo(d.id, CL_DEVICE_PROFILE, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d.id, CL_DEVICE_PROFILE, size[], result, C_NULL)
        bs = CLString(result)
        return bs
    end

    version(d::Device) = begin
        size = Ref{Csize_t}()
        clGetDeviceInfo(d.id, CL_DEVICE_VERSION, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d.id, CL_DEVICE_VERSION, size[], result, C_NULL)
        bs = CLString(result)
        return bs
    end

    driver_version(d::Device) = begin
        size = Ref{Csize_t}()
        clGetDeviceInfo(d.id, CL_DRIVER_VERSION, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d.id, CL_DRIVER_VERSION, size[], result, C_NULL)
        bs = CLString(result)
        return string(replace(bs, r"\s+" => " "))
    end

    extensions(d::Device) = begin
        size = Ref{Csize_t}()
        clGetDeviceInfo(d.id, CL_DEVICE_EXTENSIONS, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d.id, CL_DEVICE_EXTENSIONS, size[], result, C_NULL)
        bs = CLString(result)
        return String[string(s) for s in split(bs)]
    end

    platform(d::Device) = begin
        result = Ref{cl_platform_id}()
        clGetDeviceInfo(d.id, CL_DEVICE_PLATFORM,
                                   sizeof(cl_platform_id), result, C_NULL)
        return Platform(result[])
    end

    name(d::Device) = begin
        size = Ref{Csize_t}()
        clGetDeviceInfo(d.id, CL_DEVICE_NAME, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d.id, CL_DEVICE_NAME,
                                   size[] * sizeof(Cchar), result, C_NULL)
        n = CLString(result)
        return string(replace(n, r"\s+" => " "))
    end

    device_type(d::Device) = begin
        result = Ref{cl_device_type}()
        clGetDeviceInfo(d.id, CL_DEVICE_TYPE,
                                   sizeof(cl_device_type), result, C_NULL)
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

    has_image_support(d::Device) = begin
        has_support = Ref{cl_bool}(CL_FALSE)
        clGetDeviceInfo(d.id, CL_DEVICE_IMAGE_SUPPORT,
                                   sizeof(cl_bool), has_support, C_NULL)
        return has_support[] == CL_TRUE
    end

    @int_info(queue_properties, CL_DEVICE_QUEUE_PROPERTIES, cl_command_queue_properties)

    has_queue_out_of_order_exec(d::Device) =
        (queue_properties(d) & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0

    has_queue_profiling(d::Device) =
        (queue_properties(d) & CL_QUEUE_PROFILING_ENABLE) != 0

    has_native_kernel(d::Device) = begin
        result = Ref{cl_device_exec_capabilities}()
        clGetDeviceInfo(d.id, CL_DEVICE_EXECUTION_CAPABILITIES,
                                   sizeof(cl_device_exec_capabilities), result, C_NULL)
        return (result[] & CL_EXEC_NATIVE_KERNEL) != 0
    end

    @int_info(vendor_id,             CL_DEVICE_VENDOR_ID,                Cuint)
    @int_info(max_compute_units,     CL_DEVICE_MAX_COMPUTE_UNITS,        Cuint)
    @int_info(max_work_item_dims,    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, Cuint)
    @int_info(max_clock_frequency,   CL_DEVICE_MAX_CLOCK_FREQUENCY,      Cuint)
    @int_info(address_bits,          CL_DEVICE_ADDRESS_BITS,             Cuint)
    @int_info(max_read_image_args,   CL_DEVICE_MAX_READ_IMAGE_ARGS,      Cuint)
    @int_info(max_write_image_args,  CL_DEVICE_MAX_WRITE_IMAGE_ARGS,     Cuint)
    @int_info(global_mem_size,       CL_DEVICE_GLOBAL_MEM_SIZE,          Culong)
    @int_info(max_mem_alloc_size,    CL_DEVICE_MAX_MEM_ALLOC_SIZE,       Culong)
    @int_info(max_const_buffer_size, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, Culong)
    @int_info(local_mem_size,        CL_DEVICE_LOCAL_MEM_SIZE,           Culong)

    has_local_mem(d::Device) = begin
        result = Ref{cl_device_local_mem_type}()
        clGetDeviceInfo(d.id, CL_DEVICE_LOCAL_MEM_TYPE,
                                   sizeof(cl_device_local_mem_type), result, C_NULL)
        return result[] == CL_LOCAL
    end

    host_unified_memory(d::Device) = begin
        result = Ref{cl_bool}(CL_FALSE)
        clGetDeviceInfo(d.id, CL_DEVICE_HOST_UNIFIED_MEMORY,
                                   sizeof(cl_bool), result, C_NULL)
        return result[] == CL_TRUE
    end

    available(d::Device) = begin
        result = Ref{cl_bool}(CL_FALSE)
        clGetDeviceInfo(d.id, CL_DEVICE_AVAILABLE,
                                   sizeof(cl_bool), result, C_NULL)
        return result[] == CL_TRUE
    end

    compiler_available(d::Device) = begin
        result = Ref{cl_bool}(CL_FALSE)
        clGetDeviceInfo(d.id, CL_DEVICE_COMPILER_AVAILABLE,
                                   sizeof(cl_bool), result, C_NULL)
        return result[] == CL_TRUE
    end

    max_work_item_size(d::Device) = begin
        dims = Ref{Cuint}()
        clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                   sizeof(Cuint), dims, C_NULL)
        result = Vector{Csize_t}(undef, dims[])
        clGetDeviceInfo(d.id, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                   sizeof(Csize_t) * dims[], result, C_NULL)
        return tuple([Int(r) for r in result]...)
    end

    @int_info(max_work_group_size, CL_DEVICE_MAX_WORK_GROUP_SIZE, Csize_t)
    @int_info(max_parameter_size, CL_DEVICE_MAX_PARAMETER_SIZE,  Csize_t)
    @int_info(profiling_timer_resolution, CL_DEVICE_PROFILING_TIMER_RESOLUTION, Csize_t)

    max_image2d_shape(d::Device) = begin
        width  = Ref{Csize_t}()
        height = Ref{Csize_t}()
        clGetDeviceInfo(d.id, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                                   sizeof(Csize_t), width,  C_NULL)
        clGetDeviceInfo(d.id, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                                   sizeof(Csize_t), height, C_NULL)
        return (width[], height[])
    end

    max_image3d_shape(d::Device) = begin
        width  = Ref{Csize_t}()
        height = Ref{Csize_t}()
        depth =  Ref{Csize_t}()
        clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_WIDTH,
                                   sizeof(Csize_t), width, C_NULL)
        clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                                   sizeof(Csize_t), height, C_NULL)
        clGetDeviceInfo(d.id, CL_DEVICE_IMAGE3D_MAX_DEPTH,
                                   sizeof(Csize_t), depth, C_NULL)
        return (width[], height[], depth[])
    end

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
