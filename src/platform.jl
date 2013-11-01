# Low Level OpenCL Platform 

immutable Platform
    id::CL_platform_id
end

Base.pointer(p::Platform) = p.id
@ocl_object_equality(Platform)

#Base.hash(p::Platform) = unsigned(p.id)
#Base.isequal(p1::Platform, p2::Platform) = Base.hash(p1) == Base.hash(p2)

Base.getindex(p::Platform, pinfo::Symbol) = info(p, pinfo)

function Base.show(io::IO, p::Platform)
    strip_extra_whitespace = r"\s+"
    platform_name = replace(p[:name], strip_extra_whitespace, " ")
    ptr_address = "0x$(hex(unsigned(Base.pointer(p)), WORD_SIZE>>2))"
    print(io, "<OpenCL.Platform '$platform_name @$ptr_address>")
end


#Base.keys(p::Platform) = [k for k in keys(info_map)]
#Base.haskey(p::Platform, s::Symbol) = begin
#    for (k, _) in info_map
#        if k == s
#            return true
#        end
#    end
#    return false
#end

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

let 
    const info_map = (Symbol => CL_platform_info)[
        :profile => CL_PLATFORM_PROFILE,
        :version => CL_PLATFORM_VERSION,
        :name    => CL_PLATFORM_NAME,
        :vendor  => CL_PLATFORM_VENDOR,
        :extensions => CL_PLATFORM_EXTENSIONS
    ]
    
    function info(p::Platform, pinfo::Symbol)
        try
            cl_info = info_map[pinfo]
            if pinfo == :extensions
                split(info(p, cl_info))
            else
                info(p, cl_info)
            end
        catch
            error("OpenCL.Platform has no info for: $pinfo")
        end
    end
end


#name(p::Platform) = info(p::Platform, CL_PLATFORM_NAME)
#vendor(p::Platform) = info(p::Platform, CL_PLATFORM_VENDOR)
#version(p::Platform) = info(p::Platform, CL_PLATFORM_VERSION)
#profile(p::Platform) = info(p::Platform, CL_PLATFORM_PROFILE)
#extensions(p::Platform) = split(info(p::Platform, CL_PLATFORM_EXTENSIONS))

function devices(p::Platform, device_type::CL_device_type)
    ndevices = Array(CL_uint, 1)
    @check api.clGetDeviceIDs(p.id, device_type, 0, C_NULL, ndevices)
    result = Array(CL_device_id, ndevices[1])
    @check api.clGetDeviceIDs(p.id, device_type, ndevices[1], result, C_NULL)
    return [Device(id) for id in result]
end

devices(p::Platform) = devices(p, CL_DEVICE_TYPE_ALL)

#TODO: shorten this with cl_device_type
function devices(p::Platform, device_type::Symbol)
    try
       if device_type == :all
            devices(p, CL_DEVICE_TYPE_ALL)
        elseif device_type == :cpu
            devices(p, CL_DEVICE_TYPE_CPU)
        elseif device_type == :gpu
            devices(p, CL_DEVICE_TYPE_GPU)
        elseif device_type == :accelerator
            devices(p, CL_DEVICE_TYPE_ACCELERATOR)
        elseif device_type == :custom
            devices(p, CL_DEVICE_TYPE_CUSTOM)
        elseif device_type == :default
            devices(p, CL_DEVICE_TYPE_DEFAULT)
        else
            error("Unknown device type: $device_type")
        end
    catch
        # device type does not exist
        return []
    end
end

