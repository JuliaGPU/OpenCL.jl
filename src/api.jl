module api

include("types.jl")

const paths = Sys.isapple() ? String["/System/Library/Frameworks/OpenCL.framework"] : String[]

import Libdl

const libopencl = Libdl.find_library(["libOpenCL", "OpenCL"], paths)
@assert libopencl != ""

function _ocl_func(func, ret_type, arg_types)
    local args_in = Symbol[Symbol("arg$i::$T")
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

Base.cconvert{T}(::Type{Ptr{CL_user_data_tag}}, obj::T) = Ref{T}(obj)
Base.unsafe_convert{T}(::Type{Ptr{CL_user_data_tag}}, ref::Ref{T}) =
    Ptr{CL_user_data_tag}(isbits(T) ? pointer_from_objref(ref) : pointer_from_objref(ref[]))

Base.cconvert(::Type{Ptr{CL_user_data_tag}}, ptr::Ptr) = ptr
Base.unsafe_convert(::Type{Ptr{CL_user_data_tag}}, ptr::Ptr) = Ptr{CL_user_data_tag}(ptr)

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
