# Low Level OpenCL Platform 

immutable Platform
    id :: CL_platform_id
end

@ocl_func(clGetPlatformIDs, (CL_uint, Ptr{CL_platform_id}, Ptr{CL_uint}))
@ocl_func(clGetPlatformInfo, (CL_platform_id, CL_platform_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

function get_platforms()
    nplatforms = Array(CL_uint, 1)
    clGetPlatformIDs(0, C_NULL, nplatforms)
    cl_platform_ids = Array(CL_platform_id, nplatforms[1])
    clGetPlatformIDs(nplatforms[1], cl_platform_ids, C_NULL)
    return [Platform(cl_platform_ids[i]) for i in 1:length(cl_platform_ids)]
end

function num_platforms()
    nplatforms = Array(CL_uint, 1)
    clGetPlatformIDs(0, C_NULL, nplatforms)
    return int(nplatforms[1])
end
 
function get_info(p::Platform, info::CL_platform_info)
    size = Array(Csize_t, 1)
    clGetPlatformInfo(p.id, info, 0, C_NULL, size)
    result = Array(CL_char, size[1])
    clGetPlatformInfo(p.id, info, size[1], result, C_NULL)
    return bytestring(convert(Ptr{CL_char}, result))
end

const info_map = (Symbol => CL_platform_info)[
        :profile => CL_PLATFORM_PROFILE,
        :version => CL_PLATFORM_VERSION,
        :name    => CL_PLATFORM_NAME,
        :vendor  => CL_PLATFORM_VENDOR,
        :extensions => CL_PLATFORM_EXTENSIONS]
platform_info(p::Platform, info::Symbol) = get_info(p, info_map[info])

name(p::Platform) = get_info(p::Platform, CL_PLATFORM_NAME)
vendor(p::Platform) = get_info(p::Platform, CL_PLATFORM_VENDOR)
version(p::Platform) = get_info(p::Platform, CL_PLATFORM_VERSION)
profile(p::Platform) = get_info(p::Platform, CL_PLATFORM_PROFILE)
extensions(p::Platform) = split(get_info(p::Platform, CL_PLATFORM_EXTENSIONS))

function get_devices(p::Platform, device_type=CL_DEVICE_TYPE_ALL)
    ndevices = Array(CL_uint, 1)
    clGetDeviceIDs(p.id, device_type, 0, C_NULL, ndevices)
    result = Array(CL_device_id, ndevices[1])
    clGetDeviceIDs(p.id, device_type, ndevices[1], result, C_NULL)
    return [Device(result[i]) for i in 1:length(result)]
end
