#=== compiler apis ===#
@ocl_func(clUnloadCompiler, CL_int, ())

#=== memory object apis ===#
@ocl_func(clCreateSubBuffer, CL_mem,
              (CL_mem, CL_mem_flags, CL_buffer_create_type, Ptr{Void}, Ptr{CL_int}))

@ocl_func(clSetMemObjectDestructorCallback, CL_int,
              (CL_mem, CL_callback, Ptr{Void}))

@ocl_func(clCreateImage2D, CL_mem,
               (CL_context, CL_mem_flags, Ptr{CL_image_format}, Csize_t, Csize_t, Csize_t,
                Ptr{Void}, Ptr{CL_int}))

@ocl_func(clCreateImage3D, CL_mem,
               (CL_context, CL_mem_flags, Ptr{CL_image_format}, Csize_t, Csize_t, Csize_t,
                Csize_t, Ptr{Void}, Ptr{CL_int}))

#=== program object apis ===#
@ocl_func(clGetProgramInfo, CL_int,
              (CL_program, CL_program_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))

#=== event object apis ===#
@ocl_func(clCreateUserEvent, CL_event,
              (CL_context, Ptr{CL_int}))

@ocl_func(clSetUserEventStatus, CL_int, (CL_event, CL_int))

@ocl_func(clSetEventCallback, CL_int,
              (CL_event, CL_int, CL_callback, CL_user_data))

#=== enqueued commands apis ===#
@ocl_func(clEnqueueReadBufferRect, CL_int,
              (CL_command_queue, CL_mem, CL_bool,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t},
               Csize_t, Csize_t, Csize_t, Csize_t,
               Ptr{Void}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueWriteBufferRect, CL_int,
              (CL_command_queue, CL_mem, CL_bool,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t},
               Csize_t, Csize_t, Csize_t, Csize_t,
               Ptr{Void}, CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueCopyBufferRect, CL_int,
              (CL_command_queue, CL_mem, CL_mem,
               Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Csize_t},
               Csize_t, Csize_t, Csize_t, Csize_t,
               CL_uint, Ptr{CL_event}, Ptr{CL_event}))

@ocl_func(clEnqueueMarker, CL_int,
               (CL_command_queue, Ptr{CL_event}))

@ocl_func(clEnqueueWaitForEvents, CL_int,
               (CL_command_queue, CL_uint, Ptr{CL_event}))

@ocl_func(clEnqueueBarrier, CL_int,
               (CL_command_queue,))

#=== extension function access ===#
@ocl_func(clGetExtensionFunctionAddress, Ptr{Void}, (Ptr{Cchar},))
