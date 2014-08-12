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
    esc(quote
        $func($(args_in...)) = ccall(($(string(func)), libopencl),
                                      $ret_type,
                                      $arg_types,
                                      $(args_in...))
    end)
end

typealias CL_callback  Ptr{Void}
typealias CL_user_data Any

include("api/opencl_10.jl")
include("api/opencl_11.jl")
include("api/opencl_12.jl")

end
