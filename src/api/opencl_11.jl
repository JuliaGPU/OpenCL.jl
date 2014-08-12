#=== memory object apis ===#
@ocl_func_1_1(clCreateSubBuffer, CL_mem,
              (CL_mem, CL_mem_flags, CL_buffer_create_type, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_1(clSetMemObjectDestructorCallback, CL_int,
              (CL_mem, CL_callback, Ptr{Void}))

#=== program object apis ===#
@ocl_func_1_1(clGetProgramInfo, CL_int,
              (CL_program, CL_program_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== event object apis ===#
@ocl_func_1_1(clCreateUserEvent, CL_event,
              (CL_context, Ptr{CL_int}))

@ocl_func_1_1(clSetUserEventStatus, CL_int, (CL_event, CL_int))

@ocl_func_1_1(clSetEventCallback, CL_int,
              (CL_event, CL_int, CL_callback, CL_user_data))

#=== enqueued commands apis ===#
@ocl_func_1_1(clEnqueueReadBufferRect, CL_int,
              (CL_command_queue, CL_mem, CL_bool,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, 
               Csize_t, Csize_t, Csize_t, Csize_t,
               Ptr{Void}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_1(clEnqueueWriteBufferRect, CL_int,
              (CL_command_queue, CL_mem, CL_bool, 
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, 
               Csize_t, Csize_t, Csize_t, Csize_t, 
               Ptr{Void}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func_1_1(clEnqueueCopyBufferRect, CL_int,
              (CL_command_queue, CL_mem, CL_mem,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t}, 
               Csize_t, Csize_t, Csize_t, Csize_t,
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

#=== extension function access ===#
@ocl_func_1_1(clGetExtensionFunctionAddress, Ptr{Void}, (Ptr{Cchar},))

#=== deprecated functions ===#
@ocl_func_1_1(clCreateImage2D, CL_mem,
               (CL_context, CL_mem_flags, Ptr{CL_image_format}, Csize_t, Csize_t, Csize_t,
                Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_1(clCreateImage3D, CL_mem,
               (CL_context, CL_mem_flags, Ptr{CL_image_format}, Csize_t, Csize_t, Csize_t,
                Csize_t, Ptr{Void}, Ptr{CL_int}))

@ocl_func_1_1(clEnqueueMarker, CL_int,
               (CL_command_queue, Ptr{CL_event}))

@ocl_func_1_1(clEnqueueWaitForEvents, CL_int,
               (CL_command_queue, CL_uint, Ptr{CL_event}))

@ocl_func_1_1(clEnqueueBarrier, CL_int, 
               (CL_command_queue,))

@ocl_func_1_1(clUnloadCompiler, CL_int, ())