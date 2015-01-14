#=== device apis ===#
@ocl_func(clCreateSubDevices, CL_int,
              (CL_device_id, CL_device_partition_property, CL_uint, Ptr{CL_device_id}, Ptr{CL_uint}))

@ocl_func(clRetainDevice, CL_int, (CL_device_id,))

@ocl_func(clReleaseDevice, CL_int, (CL_device_id,))

#=== memory object apis ===#
@ocl_func(clCreateImage, CL_mem,
              (CL_context, CL_mem_flags, CL_image_format, CL_image_desc, Ptr{Void}, Ptr{CL_int}))

#=== program object apis ===#
@ocl_func(clCreateProgramWithBuiltInKernels, CL_program,
              (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Cchar}, Ptr{CL_int}))

@ocl_func(clCompileProgram, CL_int,
              (CL_program, CL_uint, Ptr{CL_device_id}, Ptr{CL_device_id}, Ptr{Cchar},
               CL_uint, Ptr{CL_program}, Ptr{Ptr{Char}}, CL_callback, Ptr{Void}))

@ocl_func(clLinkProgram, CL_program,
              (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Cchar}, CL_uint,
               CL_callback, Ptr{Void}, Ptr{CL_int}))

@ocl_func(clUnloadPlatformCompiler, CL_int, (CL_platform_id,))

#=== kernel object apis ===#
@ocl_func(clGetKernelArgInfo, CL_int,
              (CL_kernel, CL_uint, CL_kernel_arg_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== enqueued commands apis ===#
@ocl_func(clEnqueueFillBuffer, CL_int,
              (CL_command_queue, CL_mem, Ptr{Void}, Csize_t, Csize_t, Csize_t,
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueFillImage, CL_int,
              (CL_command_queue, CL_mem, Ptr{Void}, Ptr{Csize_t}, Ptr{Csize_t},
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueMigrateMemObjects, CL_int,
              (CL_command_queue, CL_uint, Ptr{CL_mem}, CL_mem_migration_flags,
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueMarkerWithWaitList, CL_int,
              (CL_command_queue, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueBarrierWithWaitList, CL_int,
              (CL_command_queue, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

#=== extension function access ===#
@ocl_func(clGetExtensionFunctionAddressForPlatform, Ptr{Void},
              (CL_platform_id, Ptr{Cchar}))

#=== opengl interop functions ===#

@ocl_func(clCreateFromGLTexture, CL_mem,
              (CL_context, CL_mem_flags, GL_enum, GL_int, GL_uint, Ptr{CL_int}))

#=== deprecation ===#

# @deprecate clGetExtensionFunctionAddress clGetExtensionFunctionAddressForPlatform

# @deprecate clCreateImage2D clCreateImage
# @deprecate clCreateImage3D clCreateImage

# @deprecate clEnqueueMarker clEnqueueMarkerWithWaitList
# @deprecate clEnqueueBarrier clEnqueueMarkerWithWaitList
# @deprecate clEnqueueWaitForEvents clEnqueueMarkerWithWaitList
# @deprecate clUnloadCompiler Nothing()

# @deprecate clCreateFromGLTexture2D clCreateFromGLTexture
# @deprecate clCreateFromGLTexture3D clCreateFromGLTexture
