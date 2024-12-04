# OpenCL.Device

struct Device <: CLObject
    id::cl_device_id
end

Base.unsafe_convert(::Type{cl_device_id}, d::Device) = d.id

function Base.show(io::IO, d::Device)
    strip_extra_whitespace = r"\s+"
    device_name = replace(d.name, strip_extra_whitespace => " ")
    platform_name = replace(d.platform.name, strip_extra_whitespace => " ")
    ptr_val = convert(UInt, pointer(d))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Device($device_name on $platform_name @$ptr_address)")
end

@inline function Base.getproperty(d::Device, s::Symbol)
    # simple string properties
    version_re = r"OpenCL (?<major>\d+)\.(?<minor>\d+)(?<vendor>.+)"
    @inline function get_string(prop)
        sz = Ref{Csize_t}()
        clGetDeviceInfo(d, prop, 0, C_NULL, sz)
        chars = Vector{Cchar}(undef, sz[])
        clGetDeviceInfo(d, prop, sz[], chars, C_NULL)
        return GC.@preserve chars unsafe_string(pointer(chars))
    end
    if s === :profile
        return get_string(CL_DEVICE_PROFILE)
    elseif s === :version
        str = get_string(CL_DEVICE_VERSION)
        m = match(version_re, str)
        if m === nothing
            error("Could not parse OpenCL version string: $str")
        end
        return strip(m["vendor"])
    elseif s === :opencl_version
        str = get_string(CL_DEVICE_VERSION)
        m = match(version_re, str)
        if m === nothing
            error("Could not parse OpenCL version string: $str")
        end
        return VersionNumber(parse(Int, m["major"]), parse(Int, m["minor"]))
    elseif s === :driver_version
        return get_string(CL_DRIVER_VERSION)
    elseif s === :name
        return get_string(CL_DEVICE_NAME)
    end

    # scalar values
    @inline function get_scalar(prop, typ)
        scalar = Ref{typ}()
        clGetDeviceInfo(d, prop, sizeof(typ), scalar, C_NULL)
        return Int(scalar[])
    end
    if s === :vendor_id
        return get_scalar(CL_DEVICE_VENDOR_ID, cl_uint)
    elseif s === :max_compute_units
        return get_scalar(CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint)
    elseif s === :max_work_item_dims
        return get_scalar(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint)
    elseif s === :max_clock_frequency
        return get_scalar(CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint)
    elseif s === :address_bits
        return get_scalar(CL_DEVICE_ADDRESS_BITS, cl_uint)
    elseif s === :max_read_image_args
        return get_scalar(CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint)
    elseif s === :max_write_image_args
        return get_scalar(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint)
    elseif s === :global_mem_size
        return get_scalar(CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong)
    elseif s === :max_mem_alloc_size
        return get_scalar(CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong)
    elseif s === :max_const_buffer_size
        return get_scalar(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong)
    elseif s === :local_mem_size
        return get_scalar(CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong)
    elseif s === :max_work_group_size
        return get_scalar(CL_DEVICE_MAX_WORK_GROUP_SIZE, Csize_t)
    elseif s === :max_parameter_size
        return get_scalar(CL_DEVICE_MAX_PARAMETER_SIZE, Csize_t)
    elseif s === :profiling_timer_resolution
        return get_scalar(CL_DEVICE_PROFILING_TIMER_RESOLUTION, Csize_t)
    end

    # boolean properties
    @inline function get_bool(prop)
        bool = Ref{cl_bool}()
        clGetDeviceInfo(d, prop, sizeof(cl_bool), bool, C_NULL)
        return bool[] == CL_TRUE
    end
    if s === :has_image_support
        return get_bool(CL_DEVICE_IMAGE_SUPPORT)
    elseif s === :has_local_mem
        return get_bool(CL_DEVICE_LOCAL_MEM_TYPE)
    elseif s === :host_unified_memory
        return get_bool(CL_DEVICE_HOST_UNIFIED_MEMORY)
    elseif s === :available
        return get_bool(CL_DEVICE_AVAILABLE)
    elseif s === :compiler_available
        return get_bool(CL_DEVICE_COMPILER_AVAILABLE)
    end

    if s == :extensions
        size = Ref{Csize_t}()
        clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetDeviceInfo(d, CL_DEVICE_EXTENSIONS, size[], result, C_NULL)
        bs = GC.@preserve result unsafe_string(pointer(result))
        return String[string(s) for s in split(bs)]
    end

    if s == :platform
        result = Ref{cl_platform_id}()
        clGetDeviceInfo(d, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), result, C_NULL)
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
        result = Vector{Csize_t}(undef, d.max_work_item_dims)
        clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(result), result, C_NULL)
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

function queue_properties(d::Device, type=:host)
    result = Ref{cl_command_queue_properties}()
    if type === :host
        clGetDeviceInfo(d, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                        sizeof(cl_command_queue_properties), result, C_NULL)
    elseif type === :device
        clGetDeviceInfo(d, CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,
                        sizeof(cl_command_queue_properties), result, C_NULL)
    else
        throw(ArgumentError("Unknown queue type: $type"))
    end
    mask = result[]

    return (;
        out_of_order_exec = mask & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE != 0,
        profiling = mask & CL_QUEUE_PROFILING_ENABLE != 0
    )
end

function exec_capabilities(d::Device)
    result = Ref{cl_device_exec_capabilities}()
    clGetDeviceInfo(d, CL_DEVICE_EXECUTION_CAPABILITIES,
                    sizeof(cl_device_exec_capabilities), result, C_NULL)
    mask = result[]

    return (;
        native_kernel = mask & CL_EXEC_NATIVE_KERNEL != 0,
    )
end

function svm_capabilities(d::Device)
    result = Ref{cl_device_svm_capabilities}()
    clGetDeviceInfo(d, CL_DEVICE_SVM_CAPABILITIES,
                    sizeof(cl_device_svm_capabilities), result, C_NULL)
    mask = result[]

    return (;
        coarse_grain_buffer = mask & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER != 0,
        fine_grain_buffer = mask & CL_DEVICE_SVM_FINE_GRAIN_BUFFER != 0,
        fine_grain_system = mask & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM != 0,
    )
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
