# Low Level OpenCL context

immutable Property
    id::Csize_t
    val::Csize_t
end

type CtxProperties #<: Associative{K, V}
    platform::Platform
    properties::Dict{Any, Property}
end

function set_platform!(ctx_props::CtxProperties, p::Platform)
  ctx_props.platform = Platform(p.id)
  ctx_props
end

function set_property!(ctx_props::CtxProperties, name, p::Property)
    ctx_properties[name] = p
    ctx_properties
end

#function properties(ctx_props::ContextProperties)
#    nprops = length(ctx_props.properties)
#    if nprops == 0
#        return
#    end
#    props = Array(CL_context_properties, (1 + 2 * nprops))
#    for (i, (prop, val)) in enumerate(ctx_prop.properties)
#        props[(i - 1) * 2 + 1] = cl_context_property(prop)
#        props[(i - 1) * 2 + 2] = cl_context_property(val)
#    end 
#    props[nprops * 2] = cl_context_property(C_NULL)
#    return props
#end

#Base.Dict(ctx_props::ContextProperties) = (Any=>Property)[k=>v for (k, v) in ctx_props.properties]

immutable Context
    id :: CL_context
end

#TODO: Clean up implementation...
function notify_ctx_error(error_info::Ptr{Cchar}, private_info::Ptr{Void},
                          cb::Csize_t, user_data::Ptr{Void})
    info = bytestring(unsafe_load(error_info))
    error("CTX Error: $info")
    return convert(Cint, 0)
end

const pfn_notify_ctx_error = cfunction(notify_ctx_error, Cint,
                                       (Ptr{Cchar}, Ptr{Void}, Csize_t, Ptr{Void}))

function clCreateContext(props::CL_context_properties,
                         ndevices::CL_uint,
                         devices::Ptr{CL_device_id},
                         pfn_notify::Ptr{Void},
                         user_data::Ptr{Void},
                         err_code::Ptr{CL_int})
    #TODO: Callbacks 
    ptf_notfiy = C_NULL
    local ctx::CL_context
    ctx = ccall((:clCreateContext, libopencl),
                CL_context,
                (CL_context_properties, CL_uint, Ptr{CL_device_id},
                 Ptr{Void}, Ptr{Void}, Ptr{CL_int}),
                props, ndevices, devices, pfn_notify, user_data, err_code)
    if err_code[1] != CL_SUCCESS
        ctx = CL_NULL
        return
    end
    return ctx
end

function Context(devices::Vector{Device}, device_type=CL_DEVICE_TYPE_DEFAULT)
    if devices
        num_devices = length(devices)
    end
end

@ocl_func(clGetContextInfo, (CL_context, CL_context_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#function properties(ctx::Context)
#    props_size = Array(Csize_t, 1)
#    clGetContextInfo(ctx.id, CL_CONTEXT_PROPERTIES, 0, C_NULL, props_size)
#    if props_size[1] == 0
#        return 
#    end
#    props = Array(CL_context_properties, props_size)
#    clGetContextInfo(ctx.id, CL_CONTEXT_PROPERTIES, props_size, props, C_NULL)
#    if props[0] != C_NULL
#        nprops = props_size // (2 * sizeof(CL_context_properties))
#    end 
#end

#TODO: wrap try finally
function free(ctx::Context)
    if ctx.id != C_NULL
        clReleaseContext(ctx.id)
    end
    ctx = nothing
end



