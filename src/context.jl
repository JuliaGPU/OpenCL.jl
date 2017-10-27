# OpenCL.Context

const _ctx_reference_count = Dict{CL_context, Int}()


function create_jl_reference!(ctx_id::CL_context)
    if haskey(_ctx_reference_count, ctx_id) # for the first jl reference, we already have a refcount of 1
        @check api.clRetainContext(ctx_id) # increase internal refcount, if creating an additional reference
    end
    refcount = get!(_ctx_reference_count, ctx_id, 0)
    _ctx_reference_count[ctx_id] = refcount + 1
    return
end
function free_jl_reference!(ctx_id::CL_context)
    if !haskey(_ctx_reference_count, ctx_id)
        error("Freeing unknown context")
    end
    refcount = _ctx_reference_count[ctx_id]
    if refcount == 0
        error("Double free of context id: ", ctx_id)
    elseif refcount == 1
        delete!(_ctx_reference_count, ctx_id)
        return
    end
    _ctx_reference_count[ctx_id] = refcount - 1
    return
end

type Context <: CLObject
    id :: CL_context
    # If created from ctx_id already, we need to increase the reference count
    # because then we give out multiple context references with multiple finalizers to the world
    # TODO should we make it in a way, that you can't overwrite it?
    function Context(ctx_id::CL_context; retain = false)
        retain && @check api.clRetainContext(ctx_id)
        if !is_ctx_id_alive(ctx_id)
            error("ctx_id not alive: ", ctx_id)
        end
        ctx = new(ctx_id)
        create_jl_reference!(ctx_id)
        finalizer(ctx, c -> begin
            retain || _deletecached!(c);
            if c.id != C_NULL
                release_ctx_id(c.id)
                free_jl_reference!(c.id)
                c.id = C_NULL
            end
        end )
        return ctx
    end
end

number_of_references(ctx::Context) = number_of_references(ctx.id)
function number_of_references(ctx_id::CL_context)
    refcounts = Ref{CL_uint}()
    @check api.clGetContextInfo(
        ctx_id, CL_CONTEXT_REFERENCE_COUNT,
        sizeof(CL_uint), refcounts, C_NULL
    )
    return refcounts[]
end

function is_ctx_id_alive(ctx_id::CL_context)
    number_of_references(ctx_id) > 0
end
function release_ctx_id(ctx_id::CL_context)
    if is_ctx_id_alive(ctx_id)
        @check api.clReleaseContext(ctx_id)
    else
        error("Double free for context: ", ctx_id)
    end
    return
end

Base.pointer(ctx::Context) = ctx.id

function Base.show(io::IO, ctx::Context)
    dev_strs = [replace(d[:name], r"\s+", " ") for d in devices(ctx)]
    devs_str = join(dev_strs, ",")
    ptr_val = convert(UInt, Base.pointer(ctx))
    ptr_address = "0x$(hex(ptr_val, Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Context(@$ptr_address on $devs_str)")
end

immutable _CtxErr
    handle :: Ptr{Void}
    err_info :: Ptr{Cchar}
    priv_info :: Ptr{Void}
    cb :: Csize_t
end

function ctx_notify_err(err_info::Ptr{Cchar}, priv_info::Ptr{Void},
                        cb::Csize_t, payload::Ptr{Void})
    ptr = convert(Ptr{_CtxErr}, payload)
    handle = unsafe_load(ptr).handle

    val = _CtxErr(handle, err_info, priv_info, cb)
    unsafe_store!(ptr, val)

    ccall(:uv_async_send, Void, (Ptr{Void},), handle)
    nothing
end


ctx_callback_ptr() = cfunction(ctx_notify_err, Void,
                               Tuple{Ptr{Cchar}, Ptr{Void}, Csize_t, Ptr{Void}})

function raise_context_error(error_info, private_info)
    throw(OpenCLException("OpenCL.Context error: $error_info"))
end


function Context(devs::Vector{Device};
                         properties=nothing,
                         callback::Union{Function, Void} = nothing)
    if isempty(devs)
        ArgumentError("No devices specified for context")
    end
    if properties !== nothing
        ctx_properties = _parse_properties(properties)
    else
        ctx_properties = C_NULL
    end

    n_devices = length(devs)
    device_ids = Vector{CL_device_id}(n_devices)
    for (i, d) in enumerate(devs)
        device_ids[i] = d.id
    end

    cb = Base.AsyncCondition()
    ctx_user_data = Ref(_CtxErr(Base.unsafe_convert(Ptr{Void}, cb), 0, 0, 0))

    err_code = Ref{CL_int}()
    ctx_id = api.clCreateContext(ctx_properties, n_devices, device_ids,
                                 ctx_callback_ptr(), ctx_user_data, err_code)
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end

    true_callback = callback == nothing ? raise_context_error : callback :: Function

    @async begin
        try
            Base.wait(cb)
            err = ctx_user_data[]
            error_info = unsafe_string(err.err_info)
            true_callback(error_info, "")
        catch
            rethrow()
        finally
            Base.close(cb)
        end
    end

    return Context(ctx_id)
end


Context(d::Device; properties=nothing, callback=nothing) =
        Context([d], properties=properties, callback=callback)

function Context(dev_type::CL_device_type;
                 properties=nothing, callback=nothing)
    if properties !== nothing
        ctx_properties = _parse_properties(properties)
    else
        ctx_properties = C_NULL
    end
    if callback !== nothing
        ctx_user_data = callback
    else
        ctx_user_data = raise_context_error
    end
    err_code = Ref{CL_int}()
    ctx_id = api.clCreateContextFromType(ctx_properties, dev_type,
                                         ctx_callback_ptr(), ctx_user_data, err_code)
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end
    return Context(ctx_id)
end

function Context(dev_type::Symbol;
                 properties=nothing, callback=nothing)
    Context(cl_device_type(dev_type),
            properties=properties, callback=callback)
end


function properties(ctx_id::CL_context)
    nbytes = Ref{Csize_t}(0)
    @check api.clGetContextInfo(ctx_id, CL_CONTEXT_PROPERTIES, 0, C_NULL, nbytes)

    # Calculate length of storage array
    # At nbytes[] the size of the properties array in bytes is stored
    # The length of the property array is then nbytes[] / sizeof(CL_context_properties)
    # Note: nprops should be odd since it requires a C_NULL terminated array
    nprops = div(nbytes[], sizeof(CL_context_properties))

    props = Vector{CL_context_properties}(nprops)
    @check api.clGetContextInfo(ctx_id, CL_CONTEXT_PROPERTIES,
                                nbytes[], props, C_NULL)
    #properties array of [key,value..., C_NULL]
    result = Any[]
    for i in 1:2:nprops
        key = props[i]
        value = i < nprops ? props[i+1] : nothing

        if key == CL_CONTEXT_PLATFORM
            push!(result, (key, Platform(cl_platform_id(value))))
        elseif key == CL_GL_CONTEXT_KHR ||
           key == CL_EGL_DISPLAY_KHR ||
           key == CL_GLX_DISPLAY_KHR ||
           key == CL_WGL_HDC_KHR ||
           key == CL_CGL_SHAREGROUP_KHR
            push!(result, (key, value))
        elseif is_apple() ? (key == CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE) : false
            push!(result, (key, value))
        elseif key == 0
            if i != nprops
                warn("Encountered OpenCL.Context property key == 0 at position $i")
            end
            break
        else
            warn("Unknown OpenCL.Context property key encountered $key")
        end
    end
    return result
end

function properties(ctx::Context)
    properties(ctx.id)
end

#Note: properties list needs to be terminated with a NULL value!
function _parse_properties(props)
    if isempty(props)
        return C_NULL
    end
    cl_props = CL_context_properties[]
    for prop_tuple in props
        if length(prop_tuple) != 2
            throw(ArgumentError("Context property tuples must be of type (key, value)"))
        end
        prop, val = prop_tuple
        push!(cl_props, cl_context_properties(prop))
        if prop == CL_CONTEXT_PLATFORM
            isa(val, Platform) && (val = val.id)
            push!(cl_props, cl_context_properties(val))
        elseif prop == CL_WGL_HDC_KHR
            push!(cl_props, cl_context_properties(val))
        elseif is_apple() ? (prop == CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE) : false
            push!(cl_props, cl_context_properties(val))
        elseif prop == CL_GL_CONTEXT_KHR ||
            prop == CL_EGL_DISPLAY_KHR ||
            prop == CL_GLX_DISPLAY_KHR ||
            prop == CL_CGL_SHAREGROUP_KHR
            push!(cl_props, cl_context_properties(val))
        else
            throw(OpenCLException("Invalid OpenCL Context property"))
        end
    end
    push!(cl_props, cl_context_properties(C_NULL))
    return cl_props
end

function num_devices(ctx::Context)
    ndevices = Ref{CL_uint}()
    @check api.clGetContextInfo(ctx.id, CL_CONTEXT_NUM_DEVICES,
                                sizeof(CL_uint), ndevices, C_NULL)
    return ndevices[]
end

function devices(ctx::Context)
    n = num_devices(ctx)
    if n == 0
        return []
    end
    dev_ids = Vector{CL_device_id}(n)
    @check api.clGetContextInfo(ctx.id, CL_CONTEXT_DEVICES,
                                n * sizeof(CL_device_id), dev_ids, C_NULL)
    return [Device(id) for id in dev_ids]
end

function create_some_context()
    if isempty(platforms())
        throw(OpenCLException("No OpenCL.Platform available"))
    end
    gpu_devices = devices(:gpu)
    if !isempty(gpu_devices)
        for dev in gpu_devices
            local ctx::Context
            try
                ctx = Context(dev)
            catch
                continue
            end
            return ctx
        end
    end
    cpu_devices = devices(:cpu)
    if !isempty(cpu_devices)
        for dev in cpu_devices
            local ctx::Context
            try
                ctx = Context(dev)
            catch
                continue
            end
            return ctx
        end
    end
    if isempty(gpu_devices) && isempty(cpu_devices)
        throw(OpenCLException("Unable to create any OpenCL.Context, no available devices"))
    else
        throw(OpenCLException("Unable to create any OpenCL.Context, no devices worked"))
    end
end
