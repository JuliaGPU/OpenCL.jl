type Program
    ptr::CL_program
end

@ocl_func(clReleaseProgram, (CL_program,))

function release!(p::Program)
    if p.ptr != C_NULL
        clReleaseProgram(p.ptr)
        p.ptr = C_NULL
    end
end

# low level api 
@ocl_func(clBuildProgram,
          (CL_program, CL_uint, Ptr{CL_device_id},Ptr{Cchar}, Ptr{Void}, Ptr{Void}))

@ocl_func(clGetProgramBuildInfo,
          (CL_program, CL_device_id, CL_program_build_info, Csize_t, Ptr{Void}, Ptr{Csize_t}))  

function clCreateProgramWithBinary(ctx::CL_context,
                                   num_devices::CL_uint,
                                   device_list::Ptr{CL_device_id},
                                   lengths::Ptr{Csize_t},
                                   binaries::Ptr{Ptr{Cchar}},
                                   binary_status::Ptr{CL_int},
                                   errcode_ret::Ptr{CL_int})
    program = ccall((:clCreateProgramWithBinary, libopencl),
                    CL_program,
                    (CL_context, CL_uint, Ptr{CL_device_id}, Ptr{Csize_t},
                     Ptr{Ptr{Cchar}}, Ptr{CL_int}, Ptr{CL_int}),
                     ctx, num_devices, device_list, lengths, binaries, binary_status, errcode_ret)
    return program
end

# high level api
function build_program(p::Program, d::Device)
end

function create_program_with_binary(ctx::Context, d::Device, binary::String)
end

#TODO: create_program_from_source()
#TODO: create_program_from_binary()

#TODO: get_info
#TODO: get_build_info()
#TODO: build(program; options=[], devices=None)
#TODO: compile(program, options=[], devices=nothing, headers=[])
#TODO: all_kernels(program)

#OpenCL 1.2
#TODO: create_program_with_built_in_kernels(ctx, devices, kernel_names)
#TODO: link_program(ctx, programs; options=[], devices=None)
#TODO: unload_platform_compiler(platform)
