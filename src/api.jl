module api

include("types.jl")

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

# Todo check macro
function __init__()
  err = 0
  # Get Platform IDs

  nplatforms = Array(CL_uint, 1)
  err = clGetPlatformIDs(0, C_NULL, nplatforms)
  if err != 0
    throw(err)
  end

  cl_platform_ids = Array(CL_platform_id, nplatforms[1])
  err = clGetPlatformIDs(nplatforms[1], cl_platform_ids, C_NULL)
  if err != 0
    throw(err)
  end

  # Map ids to version strings
  # Version string matcher = 
  matcher = r"^OpenCL ([0-9]+)\.([0-9]+) .*$"
  const CL_PLATFORM_VERSION = cl_uint(0x0901)

  versions = map(cl_platform_ids) do id 
    nbytes = Csize_t[0]
    err = clGetPlatformInfo(id, CL_PLATFORM_VERSION, 0, C_NULL, nbytes)
    if err != 0
      throw(err)
    end

    result = Array(CL_char, div(nbytes[1], sizeof(CL_char)))
    err = clGetPlatformInfo(id, CL_PLATFORM_VERSION, nbytes[1], result, C_NULL)
    if err != 0
      throw(err)
    end

    version = bytestring(convert(Ptr{CL_char}, result))

    mg = match(matcher, version) 
    if mg == nothing
        error("Platform $(p[:name]) returns non conformat platform string: $(ver)")
    end
    return VersionNumber(int(mg.captures[1]), int(mg.captures[2]))
  end

  global const OPENCL_VERSION = maximum(versions)
  
  if OPENCL_VERSION == v"1.1"
    @loadApi "11"

  elseif OPENCL_VERSION == v"1.2"
    @loadApi "11" "12"
  end
end
end
