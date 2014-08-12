#=== device apis ===#
@ocl_func_1_2(clCreateSubDevices, CL_int,
              (CL_device_id, CL_device_partition_property, CL_uint, Ptr{CL_device_id}, Ptr{CL_uint}))

@ocl_func_1_2(clRetainDevice, CL_int, (CL_device_id,))

@ocl_func_1_2(clReleaseDevice, CL_int, (CL_device_id,))

#=== memory object apis ===#
@ocl_func_1_2(clCreateImage, CL_mem,
              (CL_context, CL_mem_flags, CL_image_format, CL_image_desc, Ptr{Void}, Ptr{CL_int}))

#=== program object apis ===#
@ocl_func_1_2(clCreateProgramWithBuiltInKernels, CL_program,
              (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Cchar}, Ptr{CL_int}))

@ocl_func_1_2(clCompileProgram, CL_int,
              (CL_program, CL_uint, Ptr{CL_device_id}, Ptr{CL_device_id}, Ptr{Cchar}, 
               CL_uint, Ptr{CL_program}, Ptr{Ptr{Char}}, CL_callback, Ptr{Void}))

@ocl_func_1_2(clLinkProgram, CL_program,
              (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Cchar}, CL_uint,
               CL_callback, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_2(clUnloadPlatformCompiler, CL_int, (CL_platform_id,))

#=== kernel object apis ===#
@ocl_func_1_2(clGetKernelArgInfo, CL_int,
              (CL_kernel, CL_uint, CL_kernel_arg_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== enqueued commands apis ===#
@ocl_func_1_2(clEnqueueFillBuffer, CL_int,
              (CL_command_queue, CL_mem, Ptr{Void}, Csize_t, Csize_t, Csize_t, 
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_2(clEnqueueFillImage, CL_int,
              (CL_command_queue, CL_mem, Ptr{Void}, Ptr{Csize_t}, Ptr{Csize_t}, 
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_2(clEnqueueMigrateMemObjects, CL_int, 
              (CL_command_queue, CL_uint, Ptr{CL_mem}, CL_mem_migration_flags,
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_2(clEnqueueMarkerWithWaitList, CL_int,
              (CL_command_queue, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_2(clEnqueueBarrierWithWaitList, CL_int, 
              (CL_command_queue, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

#=== extension function access ===#
@ocl_func_1_2(clGetExtensionFunctionAddressForPlatform, Ptr{Void},
              (CL_platform_id, Ptr{Cchar}))

#=== deprecation ===#

@deprecate clGetExtensionFunctionAddress clGetExtensionFunctionAddressForPlatform

@deprecate clCreateImage2D clCreateImage
@deprecate clCreateImage3D clCreateImage

@deprecate clEnqueueMarker clEnqueueMarkerWithWaitList
@deprecate clEnqueueBarrier clEnqueueMarkerWithWaitList
@deprecate clEnqueueWaitForEvents clEnqueueMarkerWithWaitList
@deprecate clUnloadCompiler Nothing()