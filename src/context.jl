# Low Level OpenCL context

type Context 
    id :: CL_context
    
    function Context(ctx_id::CL_context)
        ctx = new(ctx_id)
        finalizer(ctx, c -> release!(c))
        return ctx 
    end
end

function release!(ctx::Context)
    if ctx.id != C_NULL
        @check api.clReleaseContext(ctx.id)
        ctx.id = C_NULL 
    end
end

#TODO: change to cl_pointer??? so it doesn't interfere with Base definition
Base.pointer(ctx::Context) = ctx.id
@ocl_object_equality(Context)

function Context(ctx_id::CL_context; retain=true)
    if retain
        @check api.clRetainContext(ctx_id)
    end
    return Context(ctx_id)
end

function Context(ds::Vector{Device}; properties=None, callback=None)
    if isempty(ds)
        error("No devices specified for context")
    end
    ctx_properties = C_NULL
    ctx_callback   = C_NULL
    ctx_user_data  = C_NULL
    if properties != None
        #TODO: properties
    end
    if callback != None
        #TODO: context
    end
    n_devices = length(devices)
    device_ids = Array(CL_device_id, n_devices)
    for (i, d) in enumerate(devices)
        device_ids[i] = d.id 
    end
    err_code = Array(CL_int, 1)
    ctx_id = api.clCreateContext(ctx_properties, n_devices, device_ids,
                                 ctx_callback, ctx_user_data, err_code)
    if err_code[1] != CL_SUCCESS
        throw(CLError(err_code[1]))
    end 
    return Context(ctx_id, retain=true)
end

function Context(device_type::CL_device_type; properties=None, callback=None)
    ctx_properties = C_NULL
    ctx_callback   = C_NULL
    ctx_user_data  = C_NULL

    if properties != None
        #TODO: properties
    end
    if callback != None
        user_data = convert(Ptr{Void}, callback)
    end
    err_code = Array(CL_int, 1)
    ctx_id = api.clCreateContextFromType(ctx_properties, dtype,
                                         ctx_callback, ctx_user_data, err_code)
    if err_code[1] != CL_SUCCESS
        throw(CLError(err_code[1]))
    end
    return Context(ctx_id, retain=true)
end

function Context(device_type::Symbol; properties=None, callback=None)
    Context(cl_device_type(device_type),
            properties=properties, callback=callback)
end 

function num_devices(ctx::Context)
    ndevices = Array(CL_uint, 1)
    @check api.clGetContextInfo(ctx.id, CL_CONTEXT_NUM_DEVICES,
                                sizeof(CL_uint), ndevices, C_NULL)
    return ndevices[1]
end

function devices(ctx::Context)
    n = num_devices(ctx)
    if n == 0
        return [] 
    end
    dev_ids = Array(CL_device_id, n)
    @check api.clGetContextInfo(ctx.id, CL_CONTEXT_DEVICES,
                                n * sizeof(CL_device_id), dev_ids, C_NULL)
    return [Device(id) for id in dev_ids]
end

#TODO: interative
function create_some_context(;interative=true)
    ocl_platforms = platforms()
    if isempty(ocl_platforms)
        error("No OpenCL platforms available")
    end
    platform = first(ocl_platforms)
    ocl_devices = devices(platform)
    if isempty(ocl_devices)
        error("No devices for platform: $platform")
    end 
    device = first(ocl_devices)
    return Context(device)
end

#immutable Property
#    id::Csize_t
#    val::Csize_t
#end

#type CtxProperties #<: Associative{K, V}
#    platform::Platform
#    properties::Dict{Any, Property}
#end

#function set_platform!(ctx_props::CtxProperties, p::Platform)
#  ctx_props.platform = Platform(p.id)
#  ctx_props
#end

#function set_property!(ctx_props::CtxProperties, name, p::Property)
#    ctx_properties[name] = p
#    ctx_properties
#end

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

#TODO: Clean up implementation...
#function notify_ctx_error(error_info::Ptr{Cchar}, private_info::Ptr{Void},
#                          cb::Csize_t, user_data::Ptr{Void})
#    info = bytestring(unsafe_load(error_info))
#    error("CTX Error: $info")
#    return convert(Cint, 0)
#end

#const pfn_notify_ctx_error = cfunction(notify_ctx_error, Cint,
#                                       (Ptr{Cchar}, Ptr{Void}, Csize_t, Ptr{Void}))

#TODO: Dig into the c ffi system here for the correct types
#function clCreateContext(props,
#                         ndevices,
#                         devices,
#                         pfn_notify,
#                         user_data,
#                         err_code)
#    ptf_notfiy = pfn_notify_ctx_error
#    local ctx::CL_context
#    ctx = ccall((:clCreateContext, libopencl),
#                CL_context,
#                (CL_context_properties, CL_uint, Ptr{CL_device_id},
#                 Ptr{Void}, Ptr{Void}, Ptr{CL_int}),
#                props, ndevices, devices, pfn_notify, user_data, err_code)
#    if err_code[1] != CL_SUCCESS
#        ctx = C_NULL
#    end
#    return ctx
#end

# TODO: Unimplemented: create context without specifing device
#function Context(devices::Vector{Device}, device_type=CL_DEVICE_TYPE_DEFAULT)
#    # TODO: context properties
#    #local ctx_props::CL_context_properties
#    ctx_props = C_NULL
#    user_data = C_NULL
#    num_devices = length(devices)
#    device_ids = Array(CL_device_id, num_devices)
#    for i in 1:num_devices
#        device_ids[i] = devices[i].id
#    end
#    err_code = Array(CL_int, 1)
#    local ctx_id::CL_context
#    ctx_id = clCreateContext(ctx_props, num_devices, device_ids,
#                             pfn_notify_ctx_error, user_data, err_code)
#    if err_code[1] != CL_SUCCESS || ctx_id == C_NULL
#        error("Error creating context")
#    end
#    return Context(ctx_id)
#end

#Context(device::Device, device_type=CL_DEVICE_TYPE_DEFAULT) = Context([device], device_type)

#@ocl_func(clGetContextInfo, (CL_context, CL_context_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

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

#@ocl_func(clReleaseContext, (CL_context,))

#TODO: interative

