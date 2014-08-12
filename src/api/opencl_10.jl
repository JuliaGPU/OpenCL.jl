#=== platform apis ===#
@ocl_func_1_0(clGetPlatformIDs, CL_int,
              (CL_uint, Ptr{CL_platform_id}, Ptr{CL_uint}))

@ocl_func_1_0(clGetPlatformInfo,
              CL_int, (CL_platform_id, CL_platform_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== device apis ===#
@ocl_func_1_0(clGetDeviceIDs, CL_int,
              (CL_platform_id, CL_device_type, CL_uint, Ptr{CL_device_id}, Ptr{CL_uint}))

@ocl_func_1_0(clGetDeviceInfo, CL_int,
              (CL_device_id, CL_device_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== context apis ===#
#TODO: pass user data as Any type 
@ocl_func_1_0(clCreateContext, CL_context,
              (Ptr{CL_context_properties}, CL_uint, Ptr{CL_device_id}, CL_callback, CL_user_data, Ptr{CL_int}))

@ocl_func_1_0(clCreateContextFromType, CL_context,
              (Ptr{CL_context_properties}, CL_device_type, CL_callback, CL_user_data, Ptr{CL_int}))

@ocl_func_1_0(clRetainContext, CL_int, (CL_context,))

@ocl_func_1_0(clReleaseContext, CL_int, (CL_context,))

@ocl_func_1_0(clGetContextInfo, CL_int,
              (CL_context, CL_context_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== command queue apis ===#
@ocl_func_1_0(clCreateCommandQueue, CL_command_queue,
              (CL_context, CL_device_id, CL_command_queue_properties, Ptr{CL_int}))

@ocl_func_1_0(clRetainCommandQueue, CL_int, (CL_command_queue,))

@ocl_func_1_0(clReleaseCommandQueue, CL_int, (CL_command_queue,))

@ocl_func_1_0(clGetCommandQueueInfo, CL_int,
              (CL_command_queue, CL_command_queue_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== memory object apis ===#
@ocl_func_1_0(clCreateBuffer, CL_mem, 
              (CL_context, CL_mem_flags, Csize_t, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_0(clRetainMemObject, CL_int, (CL_mem,))

@ocl_func_1_0(clReleaseMemObject, CL_int, (CL_mem,))

@ocl_func_1_0(clGetSupportedImageFormats, CL_int,
              (CL_context, CL_mem_flags, CL_mem_object_type, CL_uint, Ptr{CL_image_format}, Ptr{CL_uint}))

@ocl_func_1_0(clGetMemObjectInfo, CL_mem,
              (CL_mem, CL_mem_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

@ocl_func_1_0(clGetImageInfo, CL_mem,
              (CL_mem, CL_image_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))
 
#=== sampler apis ===#
@ocl_func_1_0(clCreateSampler, CL_sampler,
              (CL_context, CL_bool, CL_addressing_mode, CL_filter_mode, Ptr{CL_int}))

@ocl_func_1_0(clRetainSampler, CL_int, (CL_sampler,))

@ocl_func_1_0(clReleaseSampler, CL_int, (CL_sampler,))

@ocl_func_1_0(clGetSamplerInfo, CL_int,
              (CL_sampler, CL_sampler_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== program object apis ===#
@ocl_func_1_0(clCreateProgramWithSource, CL_program,
              (CL_context, CL_uint, Ptr{Ptr{Cchar}}, Ptr{Csize_t}, Ptr{CL_int}))

@ocl_func_1_0(clCreateProgramWithBinary, CL_program,
              (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Csize_t}, 
               Ptr{Ptr{Cuchar}}, Ptr{CL_int}, Ptr{CL_int}))

@ocl_func_1_0(clRetainProgram, CL_int, (CL_program,))

@ocl_func_1_0(clReleaseProgram, CL_int, (CL_program,))

@ocl_func_1_0(clBuildProgram, CL_int,
              (CL_program, CL_uint, Ptr{CL_device_id}, Ptr{Cchar}, CL_callback, Ptr{Void}))

@ocl_func_1_0(clGetProgramBuildInfo, CL_int,
              (CL_program, CL_device_id, CL_program_build_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== kernel object apis ===#
@ocl_func_1_0(clCreateKernel, CL_kernel,
              (CL_program, Ptr{Cchar}, Ptr{CL_int}))

@ocl_func_1_0(clCreateKernelsInProgram, CL_int,
              (CL_program, CL_uint, Ptr{CL_kernel}, Ptr{CL_uint}))

@ocl_func_1_0(clRetainKernel, CL_int, (CL_kernel,))

@ocl_func_1_0(clReleaseKernel, CL_int, (CL_kernel,))

@ocl_func_1_0(clSetKernelArg, CL_int,
              (CL_kernel, CL_uint, Csize_t, Ptr{Void}))

@ocl_func_1_0(clGetKernelInfo, CL_int,
              (CL_kernel, CL_kernel_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

@ocl_func_1_0(clGetKernelWorkGroupInfo, CL_int,
              (CL_kernel, CL_device_id, CL_kernel_work_group_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== event object apis ===#
@ocl_func_1_0(clWaitForEvents, CL_int,
              (CL_uint, Ptr{CL_event_info}))

@ocl_func_1_0(clGetEventInfo, CL_int,
              (CL_event, CL_event_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

@ocl_func_1_0(clRetainEvent, CL_int, (CL_event,))

@ocl_func_1_0(clReleaseEvent, CL_int, (CL_event,))

#=== profiling apis ===#
@ocl_func_1_0(clGetEventProfilingInfo, CL_int, 
              (CL_event, CL_profiling_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== flush and finish apis ===#
@ocl_func_1_0(clFlush, CL_int, (CL_command_queue,))

@ocl_func_1_0(clFinish, CL_int, (CL_command_queue,))

#=== enqueued commands apis ===#
@ocl_func_1_0(clEnqueueReadBuffer, CL_int, 
              (CL_command_queue, CL_mem, CL_bool, Csize_t, Csize_t, Ptr{Void},
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueWriteBuffer, CL_int,
              (CL_command_queue, CL_mem, CL_bool, 
               Csize_t, Csize_t, Ptr{Void}, CL_uint, 
               Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueCopyBuffer, CL_int,
              (CL_command_queue, CL_mem, CL_mem,
               Csize_t, Csize_t, Csize_t, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueReadImage, CL_int,
              (CL_command_queue, CL_mem, CL_bool,
               Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Csize_t,
               Ptr{Void}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueWriteImage, CL_int,
              (CL_command_queue, CL_mem, CL_bool, Ptr{Csize_t}, Ptr{Csize_t},
               Csize_t, Csize_t, Ptr{Void}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueCopyImage, CL_int,
              (CL_command_queue, CL_mem, CL_mem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t},
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueCopyImageToBuffer, CL_int, 
               (CL_command_queue, CL_mem, CL_mem, Ptr{Csize_t}, Ptr{Csize_t}, 
                Csize_t, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueCopyBufferToImage, CL_int,
               (CL_command_queue, CL_mem, CL_mem, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, 
                CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueMapBuffer, Ptr{Void}, 
              (CL_command_queue, CL_mem, CL_bool, CL_map_flags, Csize_t, Csize_t,
               CL_uint, Ptr{CL_event}, Ptr{CL_event}, Ptr{CL_int}))

@ocl_func_1_0(clEnqueueMapImage, Ptr{Void},
              (CL_command_queue, CL_mem, CL_bool, CL_map_flags,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, 
               CL_uint, Ptr{CL_event}, Ptr{CL_event}, Ptr{CL_int}))

@ocl_func_1_0(clEnqueueUnmapMemObject, CL_int,
              (CL_command_queue, CL_mem, Ptr{Void}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueNDRangeKernel, CL_int, 
              (CL_command_queue, CL_kernel, CL_uint,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t},
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))
              
@ocl_func_1_0(clEnqueueTask, CL_int,
              (CL_command_queue, CL_kernel, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_0(clEnqueueNativeKernel, CL_int, 
              (CL_command_queue, Ptr{Void}, Csize_t, CL_uint, 
               Ptr{CL_mem}, Ptr{Ptr{Void}}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))