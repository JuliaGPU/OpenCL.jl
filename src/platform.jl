# OpenCL.Platform

immutable Platform <: CLObject
    id::CL_platform_id
end

Base.pointer(p::Platform) = p.id

Base.getindex(p::Platform, pinfo::Symbol) = info(p, pinfo)

function Base.show(io::IO, p::Platform)
    strip_extra_whitespace = r"\s+"
    platform_name = replace(p[:name], strip_extra_whitespace, " ")
    ptr_address = "0x$(hex(unsigned(Base.pointer(p)), WORD_SIZE>>2))"
    print(io, "OpenCL.Platform('$platform_name' @$ptr_address)")
end

function platforms()
    nplatforms = Array(CL_uint, 1)
    @check api.clGetPlatformIDs(0, C_NULL, nplatforms)
    cl_platform_ids = Array(CL_platform_id, nplatforms[1])
    @check api.clGetPlatformIDs(nplatforms[1], cl_platform_ids, C_NULL)
    return [Platform(id) for id in cl_platform_ids]
end

function num_platforms()
    nplatforms = Array(CL_uint, 1)
    @check api.clGetPlatformIDs(0, C_NULL, nplatforms)
    return int(nplatforms[1])
end

function info(p::Platform, pinfo::CL_platform_info)
    size = Array(Csize_t, 1)
    @check api.clGetPlatformInfo(p.id, pinfo, 0, C_NULL, size)
    result = Array(CL_char, size[1])
    @check api.clGetPlatformInfo(p.id, pinfo, size[1], result, C_NULL)
    return bytestring(convert(Ptr{CL_char}, result))
end


let info_map = @compat Dict{Symbol, CL_platform_info}(
        :profile => CL_PLATFORM_PROFILE,
        :version => CL_PLATFORM_VERSION,
        :name    => CL_PLATFORM_NAME,
        :vendor  => CL_PLATFORM_VENDOR,
        :extensions => CL_PLATFORM_EXTENSIONS
    )
    
    function info(p::Platform, pinfo::Symbol)
        try
            cl_info = info_map[pinfo]
            if pinfo == :extensions
                split(info(p, cl_info))
            else
                info(p, cl_info)
            end
        catch err
            if isa(err, KeyError)
                throw(ArgumentError("OpenCL.Platform has no info for: $pinfo"))
            else
                throw(err)
            end
        end
    end
end

function devices(p::Platform, dtype::CL_device_type)
    try 
        ndevices = Array(CL_uint, 1)
        @check api.clGetDeviceIDs(p.id, dtype, 0, C_NULL, ndevices)
        if ndevices[1] == 0
            return []
        end
        result = Array(CL_device_id, ndevices[1])
        @check api.clGetDeviceIDs(p.id, dtype, ndevices[1], result, C_NULL)
        return [Device(id) for id in result]
    catch err
        if err.desc == :CL_DEVICE_NOT_FOUND || err.code == -1
            return []
        else
            throw(err)
        end
    end
end

devices(p::Platform) = devices(p, CL_DEVICE_TYPE_ALL)

function devices(p::Platform, dtype::Symbol)
    devices(p, cl_device_type(dtype))
end

function devices(dtype::CL_device_type)
    devs = Device[]
    for platform in platforms()
        append!(devs, devices(platform, dtype))
    end
    return devs
end
devices(dtype::Symbol) = devices(cl_device_type(dtype))

function devices()
    devs = Device[]
    for platform in platforms()
        append!(devs, devices(platform))
    end
    return devs
end

has_device_type(p::Platform, dtype) = length(devices(p, dtype)) > 0

available_devices(p::Platform, dtype::Symbol) = filter(d -> info(d, :available),  devices(p, dtype))
available_devices(p::Platform) = available_devices(p, :all)
