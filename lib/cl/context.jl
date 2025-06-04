# OpenCL.Context

mutable struct Context <: CLObject
    const id::cl_context

    # If created from ctx_id already, we need to increase the reference count
    # because then we give out multiple context references with multiple finalizers to the world
    # TODO should we make it in a way, that you can't overwrite it?
    function Context(ctx_id::cl_context; retain::Bool=false)
        ctx = new(ctx_id)
        retain && clRetainContext(ctx)
        finalizer(clReleaseContext, ctx)
        return ctx
    end
end

Base.unsafe_convert(::Type{cl_context}, ctx::Context) = ctx.id

function Base.show(io::IO, ctx::Context)
    dev_strs = [replace(d.name, r"\s+" => " ") for d in ctx.devices]
    devs_str = join(dev_strs, ",")
    ptr_val = convert(UInt, pointer(ctx))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Context(@$ptr_address on $devs_str)")
end

struct _CtxErr
    handle::Ptr{Nothing}
    err_info::Ptr{Cchar}
    priv_info::Ptr{Nothing}
    cb::Csize_t
end

const io_lock = ReentrantLock()
function log_error(message...)
    @async begin
        lock(stderr)
        lock(io_lock)
        print(stderr, string(message..., "\n"))
        unlock(io_lock)
        unlock(stderr)
    end
end

function ctx_notify_err(
        err_info::Ptr{Cchar}, priv_info::Ptr{Nothing},
        cb::Csize_t, func::Ptr{Nothing}
    )
    ccall(func, Nothing, (Ptr{Cchar}, Ptr{Nothing}, Csize_t), err_info, priv_info, cb)
    return
end


ctx_callback_ptr() = @cfunction(ctx_notify_err, Nothing,
                                (Ptr{Cchar}, Ptr{Nothing}, Csize_t, Ptr{Nothing}))

function raise_context_error(err_info, private_info, cb)
    log_error("OpenCL Error: | ", unsafe_string(err_info), " |")
    return
end

function Context(devs::Vector{Device};
                 properties=nothing,
                 callback::Union{Function, Nothing} = nothing)
    if isempty(devs)
        ArgumentError("No devices specified for context")
    end
    if properties !== nothing
        ctx_properties = encode_properties(properties)
    else
        ctx_properties = C_NULL
    end

    n_devices = length(devs)
    device_ids = Vector{cl_device_id}(undef, n_devices)
    for (i, d) in enumerate(devs)
        device_ids[i] = d.id
    end

    err_code = Ref{Cint}()
    payload = callback === nothing ? raise_context_error : callback
    f_ptr = @cfunction($payload, Nothing, (Ptr{Cchar}, Ptr{Nothing}, Csize_t))
    ctx_id = clCreateContext(
        ctx_properties, n_devices, device_ids,
        ctx_callback_ptr(), f_ptr, err_code)
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end
    return Context(ctx_id)
end


Context(d::Device; properties=nothing, callback=nothing) =
    Context([d], properties=properties, callback=callback)

function Context(dev_type; properties = nothing, callback = nothing)
    if properties !== nothing
        ctx_properties = encode_properties(properties)
    else
        ctx_properties = C_NULL
    end
    if callback !== nothing
        ctx_user_data_cb = callback
    else
        ctx_user_data_cb = raise_context_error
    end
    err_code = Ref{Cint}()
    ctx_user_data = @cfunction($ctx_user_data_cb, Nothing, (Ptr{Cchar}, Ptr{Nothing}, Csize_t))
    ctx_id = clCreateContextFromType(ctx_properties, dev_type,
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

function Base.getproperty(ctx::Context, s::Symbol)
    if s == :num_devices
        ndevices = Ref{Cuint}()
        clGetContextInfo(ctx, CL_CONTEXT_NUM_DEVICES, sizeof(Cuint), ndevices, C_NULL)
        return Int(ndevices[])
    elseif s == :devices
        n = ctx.num_devices
        if n == 0
            return Device[]
        end
        dev_ids = Vector{cl_device_id}(undef, n)
        clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sizeof(dev_ids), dev_ids, C_NULL)
        return [Device(id) for id in dev_ids]
    elseif s == :properties
        nbytes = Ref{Csize_t}(0)
        clGetContextInfo(ctx, CL_CONTEXT_PROPERTIES, 0, C_NULL, nbytes)

        # Calculate length of storage array
        # At nbytes[] the size of the properties array in bytes is stored
        # The length of the property array is then nbytes[] / sizeof(cl_context_properties)
        # Note: nprops should be odd since it requires a C_NULL terminated array
        nprops = div(nbytes[], sizeof(cl_context_properties))

        props = Vector{cl_context_properties}(undef, nprops)
        clGetContextInfo(ctx, CL_CONTEXT_PROPERTIES, nbytes[], props, C_NULL)
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
            elseif key == 0
                if i != nprops
                    @warn("Encountered OpenCL.Context property key == 0 at position $i")
                end
                break
            else
                @warn("Unknown OpenCL.Context property key encountered $key")
            end
        end
        return result
    elseif s == :reference_count
        refcount = Ref{Cuint}()
        clGetContextInfo(ctx, CL_CONTEXT_REFERENCE_COUNT, sizeof(Cuint), refcount, C_NULL)
        return Int(refcount[])
    else
        return getfield(ctx, s)
    end
end

function encode_properties(props)
    isempty(props) && return C_NULL

    cl_props = cl_context_properties[]
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
        elseif Sys.isapple() && prop == CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE
            push!(cl_props, cl_context_properties(val))
        elseif prop == CL_GL_CONTEXT_KHR ||
            prop == CL_EGL_DISPLAY_KHR ||
            prop == CL_GLX_DISPLAY_KHR ||
            prop == CL_CGL_SHAREGROUP_KHR
            push!(cl_props, cl_context_properties(val))
        else
            throw(OpenCLException("Invalid OpenCL context property '$prop'"))
        end
    end

    # terminate with NULL
    push!(cl_props, cl_context_properties(C_NULL))

    return cl_props
end

