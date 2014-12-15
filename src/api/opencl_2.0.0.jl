#=== memory command queue apis ===#
@ocl_func(clCreateCommandQueueWithProperties, CL_command_queue,
              (CL_context, CL_device_id, CL_queue_properties, Ptr{CL_int}))

#=== memory object apis ===#
@ocl_func(clCreatePipe, CL_mem,
              (CL_context, CL_mem_flags, CL_uint, CL_uint, Ptr{CL_pipe_properties}, CL_int))

@ocl_func(clGetPipeInfo, CL_int,
              (CL_mem, CL_pipe_info, Csize_t, Ptr{Void}, Ptr{Csize_t})

#=== SVM Allocation API ===#
@ocl_func(clSVMAlloc, Ptr{Void},
              (CL_context, CL_svm_mem_flags, Csize_t, CL_uint))

@ocl_func(clSVMFree, Void,
              (CL_context, Ptr{Void}))

#=== sampler apis ===#

@ocl_func(clCreateSamplerWithProperties, CL_sampler,
              (CL_context, Ptr{CL_sampler_properties}, Ptr{CL_int}))

#=== kernel object apis ===#
@ocl_func(clSetKernelArgSVMPointer, CL_int,
              (CL_kernel, CL_uint, Ptr{Void}))

@ocl_func(clSetKernelExecInfo, CL_int,
              (CL_kernel, CL_kernel_exec_info, Csize_t, Ptr{Void}))

#=== Enqueued Commands APIs ===#
@ocl_func(clEnqueueSVMFree, CL_int,
               (CL_command_queue, CL_uint, Ptr{Ptr{Void}}, Ptr{Void}, Prt{Void},
                CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueSVMMemcpy, CL_int
               (CL_command_queue, CL_bool, Ptr{Void}, Ptr{Void}, Csize_t,
                cl_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueSVMMemFill, CL_int,
               (CL_command_queue, Ptr{Void}, Ptr{Void}, Csize_t, Csize_t,
                CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueSVMMap, CL_int,
               (CL_command_queue, CL_bool, CL_map_flags, Ptr{Void}, Csize_t,
                Cl_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueSVMUnmap, CL_int, 
              (CL_command_queue, Ptr{Void}, CL_uint, Ptr{Cl_event}, Ptr{CL_event}))

#=== deprecation ===#

# @deprecate clCreateCommandQueue clCreateCommandQueueWithProperties
# @deprecate clCreateSampler clCreateSamplerWithProperties
# @deprecate clEnqueueTask