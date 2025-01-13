# OpenCL.Platform

struct Platform <: CLObject
    id::cl_platform_id
end

Base.unsafe_convert(::Type{cl_platform_id}, p::Platform) = p.id

function Base.getproperty(p::Platform, s::Symbol)
    # simple string properties
    version_re = r"OpenCL (?<major>\d+)\.(?<minor>\d+)(?<vendor>.+)"
    @inline function get_string(prop)
        sz = Ref{Csize_t}()
        clGetPlatformInfo(p, prop, 0, C_NULL, sz)
        chars = Vector{Cchar}(undef, sz[])
        clGetPlatformInfo(p, prop, sz[], chars, C_NULL)
        return GC.@preserve chars unsafe_string(pointer(chars))
    end
    if s === :profile
        return get_string(CL_PLATFORM_PROFILE)
    elseif s === :version
        str = get_string(CL_PLATFORM_VERSION)
        m = match(version_re, str)
        if m === nothing
            error("Could not parse OpenCL version string: $str")
        end
        return strip(m["vendor"])
    elseif s === :opencl_version
        str = get_string(CL_PLATFORM_VERSION)
        m = match(version_re, str)
        if m === nothing
            error("Could not parse OpenCL version string: $str")
        end
        return VersionNumber(parse(Int, m["major"]), parse(Int, m["minor"]))
    elseif s === :name
        return get_string(CL_PLATFORM_NAME)
    elseif s === :vendor
        return get_string(CL_PLATFORM_VENDOR)
    end

    if s == :extensions
        size = Ref{Csize_t}()
        clGetPlatformInfo(p, CL_PLATFORM_EXTENSIONS, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetPlatformInfo(p, CL_PLATFORM_EXTENSIONS, size[], result, C_NULL)
        return GC.@preserve result split(unsafe_string(pointer(result)))
    end

    return getfield(p, s)
end

function Base.show(io::IO, p::Platform)
    strip_extra_whitespace = r"\s+"
    platform_name = replace(p.name, strip_extra_whitespace => " ")
    ptr_val = convert(UInt, pointer(p))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Platform('$platform_name' @$ptr_address)")
end

function platforms()
    nplatforms = Ref{Cuint}()
    res = unchecked_clGetPlatformIDs(0, C_NULL, nplatforms)
    if res == CL_PLATFORM_NOT_FOUND_KHR || nplatforms[] == 0
        return Platform[]
    elseif res != CL_SUCCESS
        throw(CLError(res))
    end
    cl_platform_ids = Vector{cl_platform_id}(undef, nplatforms[])
    clGetPlatformIDs(nplatforms[], cl_platform_ids, C_NULL)
    return [Platform(id) for id in cl_platform_ids]
end

function num_platforms()
    nplatforms = Ref{Cuint}()
    clGetPlatformIDs(0, C_NULL, nplatforms)
    return Int(nplatforms[])
end

function devices(p::Platform, dtype)
    ndevices = Ref{Cuint}()
    ret = unchecked_clGetDeviceIDs(p, dtype, 0, C_NULL, ndevices)
    if ret == CL_DEVICE_NOT_FOUND || ndevices[] == 0
        return Device[]
    elseif ret != CL_SUCCESS
        throw(CLError(ret))
    end
    result = Vector{cl_device_id}(undef, ndevices[])
    clGetDeviceIDs(p, dtype, ndevices[], result, C_NULL)
    return Device[Device(id) for id in result]
end

function default_device(p::Platform)
    devs = devices(p, CL_DEVICE_TYPE_DEFAULT)
    isempty(devs) && return nothing
    # XXX: clGetDeviceIDs documents CL_DEVICE_TYPE_DEFAULT should only return one device,
    #      but it's been observed to return multiple devices on some platforms...
    return first(devs)
end

devices(p::Platform) = devices(p, CL_DEVICE_TYPE_ALL)

function devices(p::Platform, dtype::Symbol)
    devices(p, cl_device_type(dtype))
end

has_device_type(p::Platform, dtype) = length(devices(p, dtype)) > 0

available_devices(p::Platform, dtype::Symbol) = filter(d -> d.available,  devices(p, dtype))
available_devices(p::Platform) = available_devices(p, :all)
