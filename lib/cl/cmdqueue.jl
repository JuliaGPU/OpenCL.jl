# OpenCL.CmdQueue

mutable struct CmdQueue <: CLObject
    const id::cl_command_queue

    function CmdQueue(q_id::cl_command_queue; retain::Bool=false)
        q = new(q_id)
        retain && clRetainCommandQueue(q)
        finalizer(clReleaseCommandQueue, q)
        return q
    end
end

Base.unsafe_convert(::Type{cl_command_queue}, q::CmdQueue) = q.id

function Base.show(io::IO, q::CmdQueue)
    ptr_val = convert(UInt, pointer(q))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.CmdQueue(@$ptr_address)")
end

function CmdQueue(prop::Symbol)
    flags = cl_command_queue_properties(0)
    if prop == :out_of_order
        flags |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    elseif prop == :profile
        flags |= CL_QUEUE_PROFILING_ENABLE
    else
        throw(ArgumentError("Only :out_of_order and :profile flags are valid, recognized flag $prop"))
    end
    return CmdQueue(flags)
end

function CmdQueue(props::NTuple{2,Symbol})
    if !(:out_of_order in props && :profile in props)
        throw(ArgumentError("Only :out_of_order and :profile flags are vaid, unrecognized flags $props"))
    end
    flags = CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    return CmdQueue(flags)
end

function CmdQueue(flags=cl_command_queue_properties(0))
    err_code = Ref{Cint}()
    queue_id = clCreateCommandQueue(context(), device(), flags, err_code)
    if err_code[] != CL_SUCCESS
        if queue_id != C_NULL
            clReleaseCommandQueue(queue_id)
        end
        throw(CLError(err_code[]))
    end
    return CmdQueue(queue_id)
end

function flush(q::CmdQueue)
    clFlush(q)
    return q
end

function finish(q::CmdQueue)
    clFinish(q)
    return q
end

function Base.getproperty(q::CmdQueue, s::Symbol)
    if s == :context
        ctx_id = Ref{cl_context}()
        clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), ctx_id, C_NULL)
        return Context(ctx_id[], retain=true)
    elseif s == :device
        dev_id = Ref{cl_device_id}()
        clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof(cl_device_id), dev_id, C_NULL)
        return Device(dev_id[])
    elseif s == :reference_count
        ref_count = Ref{Cuint}()
        clGetCommandQueueInfo(q, CL_QUEUE_REFERENCE_COUNT, sizeof(Cuint), ref_count, C_NULL)
        return Int(ref_count[])
    elseif s == :properties
        props = Ref{cl_command_queue_properties}()
        clGetCommandQueueInfo(q, CL_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties),
                              props, C_NULL)
        return props[]
    else
        return getfield(q, s)
    end
end

context(queue::CmdQueue) = queue.context
