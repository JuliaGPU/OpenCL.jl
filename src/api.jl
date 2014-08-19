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

macro ocl_func(func, ret_type, arg_types)
    local args_in = Symbol[symbol("arg$i::$T")
                           for (i, T) in enumerate(arg_types.args)]
    @eval begin
        $func($(args_in...)) = ccall(($(string(func)), libopencl), 
                                            $ret_type,
                                            $arg_types,
                                            $(args_in...))
    end
end

macro loadApi(versions...)
  for version in versions
    include("api/opencl_$(version).jl")
  end
end

typealias CL_callback  Ptr{Void}
typealias CL_user_data Any 

@loadApi "10"

include("error.jl")
include("macros.jl")
include("platform.jl")

function parse_version(version_string)
    mg = match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
    if mg == nothing
        error("Non conform version string: $(ver)")
    end
    return VersionNumber(int(mg.captures[1]), int(mg.captures[2]))
end

# Todo check macro
function __init__()
  versions = map(platforms()) do platform
    parse_version(platform[:version])
  end

  global const OPENCL_VERSION = maximum(versions)

  if OPENCL_VERSION == v"1.1"
    @loadApi "11"

  elseif OPENCL_VERSION == v"1.2"
    @loadApi "11" "12"
  end
end
end
