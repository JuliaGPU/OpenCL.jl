module api

include("types.jl")

import OpenCL_jll

const libopencl = OpenCL_jll.libopencl

function _ocl_func(func, ret_type, arg_types)
    local args_in = Symbol[Symbol("arg$i")
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

const CL_callback  = Ptr{Nothing}

abstract type CL_user_data_tag end
const CL_user_data = Ptr{CL_user_data_tag}

Base.cconvert(::Type{Ptr{CL_user_data_tag}}, obj::T) where {T} = Ref{T}(obj)

Base.unsafe_convert(P::Type{Ptr{CL_user_data_tag}}, ptr::Ref) = P(Base.unsafe_convert(Ptr{Cvoid}, ptr))
Base.unsafe_convert(P::Type{Ptr{CL_user_data_tag}}, ptr::Ptr) = P(Base.unsafe_convert(Ptr{Cvoid}, ptr))

include("api/opencl_1.0.0.jl")
include("api/opencl_1.1.0.jl")
include("api/opencl_1.2.0.jl")
include("api/opencl_2.0.0.jl")

function parse_version(version_string)
    mg = match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
    if mg === nothing
        error("Non conforming version string: $(ver)")
    end
    return VersionNumber(parse(Int, mg.captures[1]),
                                 parse(Int, mg.captures[2]))
end

end
