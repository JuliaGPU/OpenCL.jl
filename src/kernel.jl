pe Kernel
end

type KernelArg
end

type LocalMemory
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
