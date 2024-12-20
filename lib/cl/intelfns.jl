ocl_extension(s) = cl.clGetExtensionFunctionAddressForPlatform(cl.platform(), s)

function clHostMemAllocINTEL(context, properties, size, alignment, errcode_ret)
	ocl_intel = ocl_extension("clHostMemAllocINTEL")

	ccall(ocl_intel, Ptr{Cvoid}, (cl.cl_context, Ptr{cl.cl_mem_properties_intel}, Csize_t, cl.cl_uint, Ptr{cl.cl_int}), context, properties, size, alignment, errcode_ret)
end

function clDeviceMemAllocINTEL(context, device, properties, size, alignment, errcode_ret)
	ocl_intel = ocl_extension("clDeviceMemAllocINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, device::cl.cl_device_id, properties::Ptr{cl.cl_mem_properties_intel}, size::Csize_t, alignment::cl.cl_uint, errcode_ret::Ptr{cl.cl_int})::Ptr{Cvoid}
end

function clSharedMemAllocINTEL(context, device, properties, size, alignment, errcode_ret)
	ocl_intel = ocl_extension("clSharedMemAllocINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, device::cl.cl_device_id, properties::Ptr{cl.cl_mem_properties_intel}, size::Csize_t, alignment::cl.cl_uint, errcode_ret::Ptr{cl.cl_int})::Ptr{Cvoid}
end

function clMemFreeINTEL(context, ptr)
	ocl_intel = ocl_extension("clMemFreeINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, ptr::Ptr{Cvoid})::cl.cl_int
end

function clMemBlockingFreeINTEL(context, ptr)
	ocl_intel = ocl_extension("clMemBlockingFreeINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, ptr::Ptr{Cvoid})::cl.cl_int
end

function clGetMemAllocInfoINTEL(context, ptr, param_name, param_value_size, param_value, param_value_size_ret)
	ocl_intel = ocl_extension("clGetMemAllocInfoINTEL")
	
    @ccall $ocl_intel(context::cl.cl_context, ptr::Ptr{Cvoid}, param_name::cl.cl_mem_info_intel, param_value_size::Csize_t, param_value::Ptr{Cvoid}, param_value_size_ret::Ptr{Csize_t})::cl.cl_int
end
