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
    quote
        function $func($(args_in...))
            ccall(($(string(func)), libopencl),
                   $ret_type,
                   $arg_types,
                   $(args_in...))
        end
    end
end

macro ocl_func(func, ret_type, arg_types)
    _ocl_func(func, ret_type, arg_types)
end

macro deprecate_ocl_func(func, ret_type, arg_types)
    local name = Expr(:quote, func)
    local expr = _ocl_func(func, ret_type, arg_types)
    @assert expr.args[2].head == :function

    local func_body = expr.args[2].args[2]
    @assert func_body.head == :block

    insert!(func_body.args, 2, 
            :(Base.depwarn(string($name," is deprecated"), $name)))
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

include("error.jl")
include("macros.jl")
include("platform.jl")

function parse_version(version_string)
    mg = match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
    if mg == nothing
        error("Non conforming version string: $(ver)")
    end
    return VersionNumber(int(mg.captures[1]), int(mg.captures[2]))
end

function __init__()
    versions = map(platforms()) do platform
        parse_version(platform[:version])
    end

    global const OPENCL_VERSION = maximum(versions)

    if OPENCL_VERSION == v"1.1"
        include_api(v"1.1")
    elseif OPENCL_VERSION == v"1.2"
        include_api(v"1.1", v"1.2")
    end
end

if VERSION < v"0.3-"
    __init__()
end

end
