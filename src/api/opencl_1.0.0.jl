#=== platform apis ===#
@ocl_func(clGetPlatformIDs, CL_int,
              (CL_uint, Ptr{CL_platform_id}, Ptr{CL_uint}))

@ocl_func(clGetPlatformInfo,
              CL_int, (CL_platform_id, CL_platform_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== device apis ===#
@ocl_func(clGetDeviceIDs, CL_int,
              (CL_platform_id, CL_device_type, CL_uint, Ptr{CL_device_id}, Ptr{CL_uint}))

@ocl_func(clGetDeviceInfo, CL_int,
              (CL_device_id, CL_device_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== context apis ===#
#TODO: pass user data as Any type
@ocl_func(clCreateContext, CL_context,
              (Ptr{CL_context_properties}, CL_uint, Ptr{CL_device_id}, CL_callback, CL_user_data, Ptr{CL_int}))

@ocl_func(clCreateContextFromType, CL_context,
              (Ptr{CL_context_properties}, CL_device_type, CL_callback, CL_user_data, Ptr{CL_int}))

@ocl_func(clRetainContext, CL_int, (CL_context,))

@ocl_func(clReleaseContext, CL_int, (CL_context,))

@ocl_func(clGetContextInfo, CL_int,
              (CL_context, CL_context_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== command queue apis ===#
@ocl_func(clCreateCommandQueue, CL_command_queue,
              (CL_context, CL_device_id, CL_command_queue_properties, Ptr{CL_int}))

@ocl_func(clRetainCommandQueue, CL_int, (CL_command_queue,))

@ocl_func(clReleaseCommandQueue, CL_int, (CL_command_queue,))

@ocl_func(clGetCommandQueueInfo, CL_int,
              (CL_command_queue, CL_command_queue_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== memory object apis ===#
@ocl_func(clCreateBuffer, CL_mem,
              (CL_context, CL_mem_flags, Csize_t, Ptr{Nothing}, Ptr{CL_int}))

@ocl_func(clRetainMemObject, CL_int, (CL_mem,))

@ocl_func(clReleaseMemObject, CL_int, (CL_mem,))

@ocl_func(clGetSupportedImageFormats, CL_int,
              (CL_context, CL_mem_flags, CL_mem_object_type, CL_uint, Ptr{CL_image_format}, Ptr{CL_uint}))

@ocl_func(clGetMemObjectInfo, CL_mem,
              (CL_mem, CL_mem_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

@ocl_func(clGetImageInfo, CL_mem,
              (CL_mem, CL_image_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== sampler apis ===#
@ocl_func(clCreateSampler, CL_sampler,
              (CL_context, CL_bool, CL_addressing_mode, CL_filter_mode, Ptr{CL_int}))

@ocl_func(clRetainSampler, CL_int, (CL_sampler,))

@ocl_func(clReleaseSampler, CL_int, (CL_sampler,))

@ocl_func(clGetSamplerInfo, CL_int,
              (CL_sampler, CL_sampler_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== program object apis ===#
@ocl_func(clCreateProgramWithSource, CL_program,
              (CL_context, CL_uint, Ptr{Ptr{Cchar}}, Ptr{Csize_t}, Ptr{CL_int}))

@ocl_func(clCreateProgramWithBinary, CL_program,
              (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Csize_t},
               Ptr{Ptr{Cuchar}}, Ptr{CL_int}, Ptr{CL_int}))

@ocl_func(clRetainProgram, CL_int, (CL_program,))

@ocl_func(clReleaseProgram, CL_int, (CL_program,))

@ocl_func(clBuildProgram, CL_int,
              (CL_program, CL_uint, Ptr{CL_device_id}, Ptr{Cchar}, CL_callback, Ptr{Nothing}))

@ocl_func(clGetProgramBuildInfo, CL_int,
              (CL_program, CL_device_id, CL_program_build_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== kernel object apis ===#
@ocl_func(clCreateKernel, CL_kernel,
              (CL_program, Ptr{Cchar}, Ptr{CL_int}))

@ocl_func(clCreateKernelsInProgram, CL_int,
              (CL_program, CL_uint, Ptr{CL_kernel}, Ptr{CL_uint}))

@ocl_func(clRetainKernel, CL_int, (CL_kernel,))

@ocl_func(clReleaseKernel, CL_int, (CL_kernel,))

@ocl_func(clSetKernelArg, CL_int,
              (CL_kernel, CL_uint, Csize_t, Ptr{Nothing}))

@ocl_func(clGetKernelInfo, CL_int,
              (CL_kernel, CL_kernel_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

@ocl_func(clGetKernelWorkGroupInfo, CL_int,
              (CL_kernel, CL_device_id, CL_kernel_work_group_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== event object apis ===#
@ocl_func(clWaitForEvents, CL_int,
              (CL_uint, Ptr{CL_event_info}))

@ocl_func(clGetEventInfo, CL_int,
              (CL_event, CL_event_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

@ocl_func(clRetainEvent, CL_int, (CL_event,))

@ocl_func(clReleaseEvent, CL_int, (CL_event,))

#=== profiling apis ===#
@ocl_func(clGetEventProfilingInfo, CL_int,
              (CL_event, CL_profiling_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))

#=== flush and finish apis ===#
@ocl_func(clFlush, CL_int, (CL_command_queue,))

@ocl_func(clFinish, CL_int, (CL_command_queue,))

#=== enqueued commands apis ===#
@ocl_func(clEnqueueReadBuffer, CL_int,
              (CL_command_queue, CL_mem, CL_bool, Csize_t, Csize_t, Ptr{Nothing},
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueWriteBuffer, CL_int,
              (CL_command_queue, CL_mem, CL_bool,
               Csize_t, Csize_t, Ptr{Nothing}, CL_uint,
               Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueCopyBuffer, CL_int,
              (CL_command_queue, CL_mem, CL_mem,
               Csize_t, Csize_t, Csize_t, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueReadImage, CL_int,
              (CL_command_queue, CL_mem, CL_bool,
               Ptr{Csize_t}, Ptr{Csize_t}, Csize_t, Csize_t,
               Ptr{Nothing}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueWriteImage, CL_int,
              (CL_command_queue, CL_mem, CL_bool, Ptr{Csize_t}, Ptr{Csize_t},
               Csize_t, Csize_t, Ptr{Nothing}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueCopyImage, CL_int,
              (CL_command_queue, CL_mem, CL_mem, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t},
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueCopyImageToBuffer, CL_int,
               (CL_command_queue, CL_mem, CL_mem, Ptr{Csize_t}, Ptr{Csize_t},
                Csize_t, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueCopyBufferToImage, CL_int,
               (CL_command_queue, CL_mem, CL_mem, Csize_t, Ptr{Csize_t}, Ptr{Csize_t},
                CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueMapBuffer, Ptr{Nothing},
              (CL_command_queue, CL_mem, CL_bool, CL_map_flags, Csize_t, Csize_t,
               CL_uint, Ptr{CL_event}, Ptr{CL_event}, Ptr{CL_int}))

@ocl_func(clEnqueueMapImage, Ptr{Nothing},
              (CL_command_queue, CL_mem, CL_bool, CL_map_flags,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t},
               CL_uint, Ptr{CL_event}, Ptr{CL_event}, Ptr{CL_int}))

@ocl_func(clEnqueueUnmapMemObject, CL_int,
              (CL_command_queue, CL_mem, Ptr{Nothing}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueNDRangeKernel, CL_int,
              (CL_command_queue, CL_kernel, CL_uint,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t},
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueTask, CL_int,
              (CL_command_queue, CL_kernel, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueNativeKernel, CL_int,
              (CL_command_queue, Ptr{Nothing}, Csize_t, CL_uint,
               Ptr{CL_mem}, Ptr{Ptr{Nothing}}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

#== opengl interop functions ==#

@ocl_func(clEnqueueAcquireGLObjects, CL_int,
              (CL_command_queue, CL_uint, Ptr{CL_mem}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueReleaseGLObjects, CL_int,
              (CL_command_queue, CL_uint, Ptr{CL_mem}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clCreateFromGLBuffer, CL_mem,
              (CL_context, CL_mem_flags, GL_uint, Ptr{CL_int}))

@ocl_func(clCreateFromGLRenderbuffer, CL_mem,
              (CL_context, CL_mem_flags, GL_uint, Ptr{CL_int}))

@ocl_func(clCreateFromGLTexture2D, CL_mem,
              (CL_context, CL_mem_flags, GL_enum, GL_int, GL_uint, Ptr{CL_int}))

@ocl_func(clCreateFromGLTexture3D, CL_mem,
              (CL_context, CL_mem_flags, GL_enum, GL_int, GL_uint, Ptr{CL_int}))

@ocl_func(clGetGLObjectInfo, CL_int,
              (CL_mem, Ptr{CL_GL_object_type}, Ptr{GL_uint}))

@ocl_func(clGetGLTextureInfo, CL_int,
              (CL_mem, CL_GL_texture_info, Csize_t, Ptr{Nothing}, Ptr{Csize_t}))
