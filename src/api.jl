module api

include("types.jl")
include("constants.jl")

@osx_only begin
    const libopencl = "/System/Library/Frameworks/OpenCL.framework/OpenCL"
end

@linux_only begin
    const libopencl = "libOpenCL"
end

@windows_only begin
    const libopencl = "OpenCL"
end

function _ocl_func(func, ret_type, arg_types)
    local args_in = Symbol[symbol("arg$i::$T")
                           for (i, T) in enumerate(arg_types.args)]
    esc(quote
        function $func($(args_in...))
            ccall(($(string(func)), libopencl),
                   $ret_type,
                   $arg_types,
                   $(args_in...))
        end
    end)
end

macro ocl_func(func, ret_type, arg_types)
    _ocl_func(func, ret_type, arg_types)
end

macro deprecate_ocl_func(func, ret_type, arg_types)
    local name = Expr(:quote, func)
    local expr = _ocl_func(func, ret_type, arg_types)
    @assert expr.args[1].args[2].head == :function
    local func_body = expr.args[1].args[2].args[2]
    @assert func_body.head == :block
    insert!(func_body.args, 2, 
            :(Base.depwarn(string($name, " is deprecated"), $name)))
    expr
end

function include_api(versions...)
    for ver in versions
        include("api/opencl_$ver.jl")
    end
end

typealias CL_callback  Ptr{Void}
typealias CL_user_data Any 

include_api(v"1.0")

function parse_version(version_string)
    mg = match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
    if mg == nothing
        error("Non conforming version string: $(ver)")
    end
    return VersionNumber(int(mg.captures[1]), int(mg.captures[2]))
end

function platform_versions()
    nplatforms = CL_uint[0]
    err = clGetPlatformIDs(0, C_NULL, nplatforms)
    err != CL_SUCCESS && error("error initializing OpenCL platforms")
    cl_platform_ids = Array(CL_platform_id, nplatforms[1])
    err = clGetPlatformIDs(nplatforms[1], cl_platform_ids, C_NULL)
    err != CL_SUCCESS && error("error initializing OpenCL platforms")
    vers = VersionNumber[]
    for pid in cl_platform_ids
        size = Csize_t[0]
        err = clGetPlatformInfo(pid, CL_PLATFORM_VERSION, 0, C_NULL, size)
        err != CL_SUCCESS && error("error initializing OpenCL platforms")
        result = Array(CL_char, size[1])
        err = clGetPlatformInfo(pid, CL_PLATFORM_VERSION, size[1], result, C_NULL)
        err != CL_SUCCESS && error("error initializing OpenCL platforms")
        push!(vers, parse_version(bytestring(convert(Ptr{CL_char}, result))))
    end
    return vers
end

const OPENCL_VERSION = maximum(platform_versions())

if OPENCL_VERSION == v"1.1"
    include_api(v"1.1")
elseif OPENCL_VERSION == v"1.2"
    include_api(v"1.1", v"1.2")
end

end
