# OpenCL.CmdQueue 

type CmdQueue 
    id::CL_command_queue

    function CmdQueue(q_id::CL_command_queue; retain=false)
        if retain
            @check api.clRetainCommandQueue(q_id)
        end
        q = new(q_id)
        finalizer(q, x -> release!(x))
        return q
    end
end 

function release!(q::CmdQueue)
    if q.id != C_NULL
        @check api.clReleaseCommandQueue(q.id)
        q.id = C_NULL
    end
end

Base.pointer(q::CmdQueue) = q.id
@ocl_object_equality(CmdQueue) 

function Base.show(io::IO, q::CmdQueue)
    ptr_address = "0x$(hex(unsigned(Base.pointer(q)), WORD_SIZE>>2))"
    print(io, "OpenCL.CmdQueue(@$ptr_address)")
end

Base.getindex(q::CmdQueue, qinfo::Symbol) = info(q, qinfo)

function CmdQueue(ctx::Context)
    devs  = devices(ctx)
    flags = cl_command_queue_properties(0) 
    if isempty(devs)
        throw(ArgumentError("OpenCL.CmdQueue Context argument does not have any devices"))
    end
    return CmdQueue(ctx, first(devs), flags)
end

function CmdQueue(ctx::Context, prop::Symbol)
    devs = devices(ctx)
    if isempty(devs)
        throw(ArgumentError("OpenCL.CmdQueue Context argument does not have any devices"))
    end
    return CmdQueue(ctx, first(devs), prop)
end

function CmdQueue(ctx::Context, props::NTuple{2, Symbol})
    devs = devices(ctx)
    if isempty(devs)
        throw(ArgumentError("OpenCL.CmdQueue Context argument does not have any devices"))
    end 
    return CmdQueue(ctx, first(devs), props)
end

function CmdQueue(ctx::Context, dev::Device)
    flags = cl_command_queue_properties(0)
    return CmdQueue(ctx, dev, flags)
end

function CmdQueue(ctx::Context, dev::Device, prop::Symbol)
    flags = cl_command_queue_properties(0)
    if prop == :out_of_order
        flags |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    elseif prop == :profile
        flags |= CL_QUEUE_PROFILING_ENABLE
    else
        throw(ArgumentError("Only :out_of_order and :profile flags are valid, recognized flag $prop"))
    end
    return CmdQueue(ctx, dev, flags)
end

function CmdQueue(ctx::Context, dev::Device, props::NTuple{2,Symbol})
    if !(:out_of_order in props && :profile in props)
        throw(ArgumentError("Only :out_of_order and :profile flags are vaid, unrecognized flags $props"))
    end
    flags = CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    return CmdQueue(ctx, dev, flags)
end

function CmdQueue(ctx::Context, dev::Device, props::CL_command_queue_properties)
    err_code = Array(CL_int, 1)
    queue_id = api.clCreateCommandQueue(ctx.id, dev.id, props, err_code)
    if err_code[1] != CL_SUCCESS
        if queue_id != C_NULL
            api.clReleaseCommandQueue(queue_id)
        end
        throw(CLError(err_code[1]))
    end
    return CmdQueue(queue_id)
end 

function flush(q::CmdQueue)
    @check api.clFlush(q.id)
    return q
end

function finish(q::CmdQueue)
    @check api.clFinish(q.id)
    return q
end

let 
    context(q::CmdQueue) = begin
        ctx_id = Array(CL_context, 1)
        @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_CONTEXT,
                                         sizeof(CL_context), ctx_id, C_NULL)
        Context(ctx_id[1], retain=true)
    end
                                          
    device(q::CmdQueue) = begin
        dev_id = Array(CL_device_id, 1)
        @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_DEVICE, 
                                         sizeof(CL_device_id), dev_id, C_NULL)
        Device(dev_id[1])
    end

    reference_count(q::CmdQueue) = begin
        ref_count = Array(CL_uint, 1)
        @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_REFERENCE_COUNT, 
                                         sizeof(CL_uint), ref_count, C_NULL)
        ref_count[1]
    end

    properties(q::CmdQueue) = begin
        props = Array(CL_command_queue_properties, 1)
        @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_PROPERTIES,
                                         sizeof(CL_command_queue_properties),
                                         props, C_NULL)
        props[1]
    end

    const info_map = (Symbol => Function)[
        :context => context,
        :device => device,
        :reference_count => reference_count,
        :properties => properties
    ]

    function info(q::CmdQueue, qinfo::Symbol)
        try
            func = info_map[qinfo]
            func(q)
        catch err
            if isa(err, KeyError)
                throw(ArgumentError("OpenCL.CmdQueue has no info for: $qinfo"))
            else
                throw(err)
            end
        end
    end
end
