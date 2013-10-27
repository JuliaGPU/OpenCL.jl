module OpenCL

# OpenCL Types 
include("types.jl")

# OpenCL Constants 
include("constants.jl")

@linux_only begin
    const libopencl = "libOpenCL"
end

macro ocl_call(func, arg_types, args...)
    quote
        _err = ccall(($func, libopencl), CL_int, $arg_types, $(args...))
        if _err != CL_SUCCESS
            error("CL ERROR: $func")
        end
    end
end

macro ocl_func(func, arg_types)
    local args_in = Symbol[symbol(string('a', i)) for i in 1:length(arg_types.args)]
    quote
        $(esc(func))($(args_in...)) = @ocl_call($(string(func)), $arg_types, $(args_in...))
    end
end


# --- OpenCL Platform --- 
include("platform.jl")

# --- OpenCL Device --- 
include("device.jl")

end # module
