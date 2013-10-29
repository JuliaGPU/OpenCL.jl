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
function clBuildProgram()
end

function clGetProgramBuildInfo()
end

function clCreateProgramWithBinary()
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
