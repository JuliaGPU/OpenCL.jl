# low level OpenCL queue

immutable Queue
    id::CL_command_queue
end 

#TODO: manage the handling of the error code in function (unify)
function clCreateCommandQueue(ctx_id::CL_context, dev_id::CL_device_id,
                              props::CL_command_queue_properties, err::Ptr{CL_int}) 
    q = ccall((:clCreateCommandQueue, libopencl),
              CL_command_queue, 
              (CL_context, CL_device_id, CL_command_queue_properties, Ptr{CL_int}),
              ctx_id, dev_id, props, err)
    err_code = unsafe_load(err)
    if err_code != CL_SUCCESS; q = C_NULL; end
    return q
end

#TODO: support queue properties
function Queue(ctx::Context, dev::Device)
    ctx_id = ctx.id
    dev_id = dev.id
    err = convert(Ptr{CL_int}, Array(CL_int, 1))
    props = cl_command_queue_properties(0)
    queue = clCreateCommandQueue(ctx_id, dev_id, props, err)
    err_code = unsafe_load(err)
    if err_code != CL_SUCCESS 
        free(queue)
        return 
    end
    return Queue(queue)
end 

@ocl_func(clGetCommandQueueInfo, (CL_command_queue, CL_command_queue_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

function device(q::Queue)
    dev_id = CL_device_id[0]
    clGetCommandQueueInfo(q.id, CL_QUEUE_DEVICE, sizeof(CL_device_id), dev_id, C_NULL)
    return Device(dev_id[1])
end

function context(q::Queue)
    ctx_id = CL_context[0]
    clGetCommandQueueInfo(q.id, CL_QUEUE_CONTEXT, sizeof(CL_context), ctx_id, C_NULL)
    return Context(ctx_id[1])
end

@ocl_func(clEnqueueBarrier, (CL_command_queue,))

#TODO: put in event.jl
#TODO: wait_for=Noen
function enqueue_barrier(q::Queue)
    clEnqueueBarrier(q.id)
end

@ocl_func(clFlush, (CL_command_queue,))

function flush(q::Queue)
    clFlush(q.id)
end

#TODO: function finish(q::Queue)

@ocl_func(clEnqueueMarker, (CL_command_queue, CL_event))

#TODO: put in event.jl
#TODO: wait_for=None
function enqueue_marker(q::Queue)
   evt_id = CL_event[0]
   clEnqueueMarker(q.id, evt_id)
   return Event(evt_id[1])
end 

@ocl_func(clReleaseCommandQueue, (CL_command_queue,))

#TODO: Better implementation
function free!(q::Queue)
    if q.id != C_NULL
        clReleaseCommandQueue(q.id)
    end
end

