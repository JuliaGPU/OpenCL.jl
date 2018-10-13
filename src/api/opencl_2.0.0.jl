#=== memory command queue apis ===#
@ocl_func(clCreateCommandQueueWithProperties, CL_command_queue,
              (CL_context, CL_device_id, CL_queue_properties, Ptr{CL_int}))

#=== memory object apis ===#
@ocl_func(clCreatePipe, CL_mem,
              (CL_context, CL_mem_flags, CL_uint, CL_uint, Ptr{CL_pipe_properties}, CL_int))

@ocl_func(clGetPipeInfo, CL_int,
              (CL_mem, CL_pipe_info, Csize_t, Ptr{Cvoid}, Ptr{Csize_t}))

#=== SVM Allocation API ===#
@ocl_func(clSVMAlloc, Ptr{Cvoid},
              (CL_context, CL_svm_mem_flags, Csize_t, CL_uint))

@ocl_func(clSVMFree, Cvoid,
              (CL_context, Ptr{Cvoid}))

#=== sampler apis ===#

@ocl_func(clCreateSamplerWithProperties, CL_sampler,
              (CL_context, Ptr{CL_sampler_properties}, Ptr{CL_int}))

#=== kernel object apis ===#
@ocl_func(clSetKernelArgSVMPointer, CL_int,
              (CL_kernel, CL_uint, Ptr{Cvoid}))

@ocl_func(clSetKernelExecInfo, CL_int,
              (CL_kernel, CL_kernel_exec_info, Csize_t, Ptr{Cvoid}))

#=== Enqueued Commands APIs ===#
@ocl_func(clEnqueueSVMFree, CL_int,
               (CL_command_queue, CL_uint, Ptr{Ptr{Cvoid}}, Ptr{Cvoid}, Ptr{Cvoid},
                CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueSVMMemcpy, CL_int,
               (CL_command_queue, CL_bool, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t,
                CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueSVMMemFill, CL_int,
               (CL_command_queue, Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Csize_t,
                CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueSVMMap, CL_int,
               (CL_command_queue, CL_bool, CL_map_flags, Ptr{Cvoid}, Csize_t,
                CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueSVMUnmap, CL_int,
              (CL_command_queue, Ptr{Cvoid}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

#=== deprecation ===#

# @deprecate clCreateCommandQueue clCreateCommandQueueWithProperties
# @deprecate clCreateSampler clCreateSamplerWithProperties
# @deprecate clEnqueueTask
