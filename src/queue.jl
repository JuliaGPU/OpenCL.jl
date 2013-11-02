# low level OpenCL queue

type CommandQueue
    id::CL_command_queue

    function CommandQueue(q_id::CL_command_queue; retain=true)
        if retain
            @check api.clRetainCommandQueue(q_id)
        end
        q = new(q_id)
        finalizer(q, cmd_q -> release!(cmd_q))
        return q
    end
end 

function release!(q::CommandQueue)
    if ctx.id != C_NULL
        @check api.clReleaseCommandQueue(q)
        q.id = C_NULL
    end
end

Base.pointer(q::CommandQueue) = q.id
@ocl_object_equality(CommandQueue) 

function Base.show(io::IO, q::CommandQueue)
    ptr_address = "0x$(hex(unsigned(Base.pointer(q)), WORD_SIZE>>2))"
    print(io, "<OpenCL.CommandQueue @$ptr_address>")
end

Base.getindex(q::CommandQueue, qinfo::Symbol) = info(q, qinfo)

function CommandQueue(ctx::Context, dev::Device; properties=None)
    ctx_id = ctx.id
    dev_id = dev.id
    err_code = Array(CL_int, 1)
    if properties == None
        props = cl_command_queue_properties(0)
    else
        props = cl_command_queue_properties(properties)
    end
    props = cl_command_queue_properties(0)
    queue_id = @check api.clCreateCommandQueue(ctx_id, dev_id, props, err_code)
    if err_code[1] != CL_SUCCESS 
        if queue_id != C_NULL
            @check api.clReleaseCommandQueue(queue_id)
        end
        throw(CLError(err_code[1]))
    end
    return CommandQueue(queue_id)
end 

function CommandQueue(ctx::Context; properties=None)
    devs = devices(ctx)
    if isempty(devs)
        error("CommandQueue context does not have any devices")
    end
    return CommandQueue(ctx, first(devs), properties=properties)
end

function device(q::CommandQueue)
    dev_id = CL_device_id[0]
    @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_DEVICE, 
                                     sizeof(CL_device_id), dev_id, C_NULL)
    return Device(dev_id[1])
end

function context(q::CommandQueue)
    ctx_id = CL_context[0]
    @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_CONTEXT, 
                                     sizeof(CL_context), ctx_id, C_NULL)
    return Context(ctx_id[1], retain=true)
end

function flush(q::CommandQueue)
    @check api.clFlush(q.id)
    return q
end

function finish(q::CommandQueue)
    @check api.clFinish(q.id)
    return q
end

let context(q::CommandQueue) = begin
        ctx_id = Array(CL_context, 1)
        @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_CONTEXT,
                                         sizeof(CL_context), ctx_id, C_NULL)
        Context(ctx_id[1])
    end
                                          
    device(q::CommandQueue) = begin
        dev_id = Array(CL_device_id)
        @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_DEVICE, 
                                         sizeof(CL_device_id), dev_id, C_NULL)
        Device(dev_id[1])
    end

    reference_count(q::CommandQueue) = begin
        ref_count = Array(CL_uint, 1)
        @check api.clGetCommandQueueInfo(q.id, CL_QUEUE_REFERENCE_COUNT, 
                                         sizeof(CL_uint), ref_count, C_NULL)
        ref_count[1]
    end

    properties(q::CommandQueue) = begin
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

    function info(q::CommandQueue, qinfo)
        try
            func = info_map[qinfo]
            func(d)
        catch err
            if isa(err, KeyError)
                error("OpenCL.CommandQueue has no info for: $qinfo") 
            else
                throw(err)
            end
        end
    end
end
