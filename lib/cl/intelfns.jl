ocl_extension(s) = cl.clGetExtensionFunctionAddressForPlatform(cl.platform(), s)

function ext_clHostMemAllocINTEL(context, properties, size, alignment, errcode_ret)
	ocl_intel = ocl_extension("clHostMemAllocINTEL")

	ccall(ocl_intel, Ptr{Cvoid}, (cl.cl_context, Ptr{cl.cl_mem_properties_intel}, Csize_t, cl.cl_uint, Ptr{cl.cl_int}), context, properties, size, alignment, errcode_ret)
end

function ext_clDeviceMemAllocINTEL(context, device, properties, size, alignment, errcode_ret)
	ocl_intel = ocl_extension("clDeviceMemAllocINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, device::cl.cl_device_id, properties::Ptr{cl.cl_mem_properties_intel}, size::Csize_t, alignment::cl.cl_uint, errcode_ret::Ptr{cl.cl_int})::Ptr{Cvoid}
end

function ext_clSharedMemAllocINTEL(context, device, properties, size, alignment, errcode_ret)
	ocl_intel = ocl_extension("clSharedMemAllocINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, device::cl.cl_device_id, properties::Ptr{cl.cl_mem_properties_intel}, size::Csize_t, alignment::cl.cl_uint, errcode_ret::Ptr{cl.cl_int})::Ptr{Cvoid}
end

function ext_clMemFreeINTEL(context, ptr)
	ocl_intel = ocl_extension("clMemFreeINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, ptr::PtrOrCLPtr{Cvoid})::cl.cl_int
end

function ext_clMemBlockingFreeINTEL(context, ptr)
	ocl_intel = ocl_extension("clMemBlockingFreeINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, ptr::PtrOrCLPtr{Cvoid})::cl.cl_int
end

function ext_clGetMemAllocInfoINTEL(context, ptr, param_name, param_value_size, param_value, param_value_size_ret)
	ocl_intel = ocl_extension("clGetMemAllocInfoINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, ptr::PtrOrCLPtr{Cvoid}, param_name::cl.cl_mem_info_intel, param_value_size::Csize_t, param_value::Ptr{Cvoid}, param_value_size_ret::Ptr{Csize_t})::cl.cl_int
end

function ext_clEnqueueMemcpyINTEL(command_queue, blocking, dst_ptr, src_ptr, size, num_events_in_wait_list, event_wait_list, event)
    ocl_intel = ocl_extension("clEnqueueMemcpyINTEL")

    @ccall $ocl_intel(command_queue::cl_command_queue, blocking::cl_bool, dst_ptr::PtrOrCLPtr{Cvoid}, src_ptr::PtrOrCLPtr{Cvoid}, size::Csize_t, num_events_in_wait_list::cl_uint, event_wait_list::Ptr{cl_event}, event::Ptr{cl_event})::cl_int
end

function ext_clEnqueueMemFillINTEL(command_queue, dst_ptr, pattern, pattern_size, size, num_events_in_wait_list, event_wait_list, event)
    ocl_intel = ocl_extension("clEnqueueMemFillINTEL")

    @ccall $ocl_intel(command_queue::cl_command_queue, dst_ptr::PtrOrCLPtr{Cvoid}, pattern::Ptr{Cvoid}, pattern_size::Csize_t, size::Csize_t, num_events_in_wait_list::cl_uint, event_wait_list::Ptr{cl_event}, event::Ptr{cl_event})::cl_int
end

function ext_clSetKernelArgMemPointerINTEL(kernel, arg_index, arg_value)
	ocl_intel = ocl_extension("clSetKernelArgMemPointerINTEL")

    @ccall $ocl_intel(kernel::cl_kernel, arg_index::cl_uint, arg_value::Ptr{Cvoid})::cl_int
end
