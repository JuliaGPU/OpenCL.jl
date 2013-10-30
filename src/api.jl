module api

@unix_only begin
    const libopencl = "libOpenCL"
end

macro ocl_func_1_0(func, ret_type, arg_types)
    local args_in = Symbol[symbol(string('a', i)) for i in 1:length(arg_types.args)]
    quote
        $(esc(func))($(args_in...)) = ccall(($(string(func)), libopencl), 
                                            $ret_type,
                                            $arg_types,
                                            $(args_in...))
    end
end

macro ocl_func_1_1(func, ret_type, arg_types)
    quote
        @ocl_func_1_0(func, ret_type, arg_types)
    end
end

macro ocl_func_1_2(func, ret_type, arg_types)
    quote
        @ocl_func_1_0(func, ret_type, arg_types)
    end
end

typealias CL_callback::Ptr{Void}

################
# platform apis
################
@ocl_func_1_0(clGetPlatformIDs, CL_int,
              (CL_uint, Ptr{CL_platform_id}, Ptr{CL_uint}))

@ocl_func_1_0(clGetPlatformInfo,
              CL_int, (CL_platform_id, CL_platform_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

##############
# device apis
##############
@ocl_func_1_0(clGetDeviceIDs, CL_int,
              (CL_platform_id, CL_device_type, CL_uint, Ptr{CL_device_id}, Ptr{CL_uint}))

@ocl_func_1_0(clGetDeviceInfo, CL_int,
              (CL_device_id, CL_device_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

@ocl_func_1_2(clCreateSubDevices, CL_int,
              (CL_device_id, CL_device_partition_property, CL_uint, Ptr{CL_device_id}, Ptr{CL_uint}))

@ocl_func_1_2(clRetainDevice, CL_int, (CL_device_id,))

@ocl_func_1_2(clReleaseDevice, CL_int, (CL_device_id,))

###############
# context apis
###############
@ocl_func_1_0(clCreateContext, CL_context,
              (CL_context_properties, CL_device_type, CL_callback, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_0(clCreateContextFromType, CL_context,
              (CL_context_properties, CL_device_type, CL_callback, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_0(clRetainContext, CL_int, (CL_context,))

@ocl_func_1_0(clReleaseContext, CL_int, (CL_context,))

@ocl_func_1_0(clContextInfo, CL_int,
              (CL_context, CL_context_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#####################
# command queue apis
#####################
@ocl_func_1_0(clCreateCommandQueue, CL_command_queue,
              (CL_context, CL_device_id, CL_command_queue_properties, Ptr{CL_int}))

@ocl_func_1_0(clRetainCommandQueue, CL_int, (CL_command_queue,))

@ocl_func_1_0(clReleaseCommandQueue, CL_int, (CL_command_queue,))

@ocl_func_1_0(clGetCommandQueueInfo, CL_int,
              (CL_command_queue, CL_command_queue_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#####################
# memory object apis
#####################
@ocl_func_1_0(clCreateBuffer, CL_mem, 
              (CL_context, CL_mem_flags, Csize_t, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_1(clCreateSubBuffer, CL_mem,
              (CL_mem, CL_mem_flags, CL_buffer_create_type, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_2(clCreateImage, CL_mem,
              (CL_context, CL_mem_flags, CL_image_format, CL_image_desc, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_0(clRetainMemObject, CL_int, (CL_mem,))

@ocl_func_1_0(clReleaseMemObject, CL_int, (CL_mem,))

@ocl_func_1_0(clGetSupportedImageFormats, CL_int,
              (CL_context, CL_mem_flags, CL_mem_object_type, CL_uint, Ptr{CL_image_format}, Ptr{CL_uint}))

@ocl_func_1_0(clGetMemObjectInfo, CL_mem,
              (CL_mem, CL_mem_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

@ocl_func_1_0(clGetImageInfo, CL_mem,
              (CL_mem, CL_image_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))
 
@ocl_func_1_1(clSetMemObjectDestructorCallback, CL_int,
              (CL_mem, CL_callback, Ptr{Void}))

###############
# sampler apis
###############
@ocl_func_1_0(clCreateSampler, CL_sampler,
              (CL_context, CL_bool, CL_addressing_mode, CL_filter_mode, Ptr{CL_int}))

@ocl_func_1_0(clRetainSampler, CL_int, (CL_sampler,))

@ocl_func_1_0(clReleaseSampler, CL_int, (CL_sampler,))

@ocl_func_1_0(clGetSamplerInfo, CL_int,
              (CL_sampler, CL_sampler_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

######################
# program object apis 
######################
@ocl_func_1_0(clCreateProgramWithSource, CL_program,
              (CL_context, CL_uint, Ptr{Ptr{Cchar}}, Ptr{Csize_t}, Ptr{CL_int}))

@ocl_func_1_0(clCreateProgramWithBinary, CL_program,
              (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Csize_t}, 
               Ptr{Ptr{Cuchar}}, Ptr{CL_int}, Ptr{CL_int}))

@ocl_func_1_2(clCreateProgramWithBuiltInKernels, CL_program,
              (CL_context, CL_uint Ptr{CL_device_id}, Ptr{Cchar}, Ptr{CL_int}))

@ocl_func_1_0(clRetainProgram, CL_int, (CL_program,))

@ocl_func_1_0(clReleaseProgram, CL_int, (CL_program,))

@ocl_func_1_0(clBuildProgram, CL_int,
              (CL_program, CL_uint, Ptr{CL_device_id}, Ptr{Cchar}, CL_callback, Ptr{Void}))

@ocl_func_1_2(clCompileProgram, CL_int,
              (CL_program, CL_uint, Ptr{CL_device_id}, Ptr{CL_device_id}, Ptr{Cchar}, 
               CL_uint, Ptr{CL_program}, Ptr{Ptr{Char}}, CL_callback, Ptr{Void}))

@ocl_func_1_2(clLinkProgram, CL_program,
              (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Cchar}, CL_uint,
               CL_callback, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_2(clUnloadPlatformCompiler, CL_int, (CL_platform_id,))

@ocl_func_1_0(clGetProgramBuildInfo, CL_int,
              (CL_program, CL_device_id, CL_program_build_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#####################
# kernel object apis
#####################


# event object apis

# profiling apis

# flush and finish apis 

# enqueued commands apis

# extension function access

end
