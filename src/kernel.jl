
type Kernel
    id :: CL_kernel

    function Kernel(k::CL_kernel, retain=false)
        if retain
            @check api.clRetainKernel(k)
        end
        kernel = new(k)
        finalizer(kernel, k -> release!(k))
    end
end

Base.pointer(k::Kernel) = k.id
@ocl_object_equality(Kernel)

function release!(k::Kernel)
    if k.id != C_NULL
        @check api.clReleaseKernel(k.id)
        k.id = C_NULL
    end
end

function Kernel(p::Program, kernel_name::String)
    err_code = Array(CL_int, 1)
    kernel_id = api.clCreateKernel(p.id, kernel_name, err_code)
    if err_code[1] != CL_SUCCESS
        throw(CLError(err_code[1]))
    end
    return Kernel(kernel_id)
end

type KernelArg
end

immutable LocalMemory
    size::Csize_t
end
#TODO: local_memory

#TODO: get_info(k::Kernel, param)
#TODO: get_work_group_info(k::Kernel, param, d::Device)
#TODO: get_arg_info(k::Kernel, idx, param)
#TODO: set_arg(k::Kernel, idx, arg)
#TODO: set_args(k::Kernel, args...)
#TODO: call(queue, global_size, local_size, *args, global_offset=None, wait_for=None)
#TODO: ?macro capture_call

#TODO: enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size,
#                              global_work_offset=None, wait_for=None)

#TODO: enqueue_task(q::Queue, k::Kernel; wait_for=None)
#TODO: enqueue_async_kernel()
