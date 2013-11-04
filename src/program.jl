type Program
    id::CL_program
end

function release!(p::Program)
    if p.id != C_NULL
        @check api.clReleaseProgram(p.id)
        p.id = C_NULL
    end
end

function Program(ctx::Context; source=Nothing, binaries=Nothing)
    program_id::CL_program
    
    if source != nothing
        byte_source = [bytestring(source)]
        err_code = Array(CL_int, 1)
        program_id = api.clCreateProgramWithSource(ctx.id, 1, byte_source, C_NULL, err_code)
        if err_code[1] != CL_SUCCESS
            throw(CLError(err_code[1]))
        end
        return Program(program_id)
    
    elseif binaries != nothing
        n_devices = length(binaries)
        device_ids = Array(CL_device_id, n_devices)
        lengths = Array(Csize_t, n_devices)
        binary_status = Array(CL_int, n_devices)
        bins = Array(Ptr{Uint8}, n_devices)
        try
            for (i, (dev, binary)) in enumerate(binaries)
                device_ids[i] = dev.id
                lengths[i] = length(binary)
                bins[i] = convert(Ptr{Uint8}, binary)
            end
            err_code = Array[1]
            program_id = api.clCreateProgramWithBinary(ctx.id, n_devices, device_ids, 
                                                       lengths, bins, binary_status, err_code)
            if err_code[1] != CL_SUCCESS
                throw(CLError(err_code[1]))
            end
            for status in binary_status
                if status != CL_SUCCESS
                    throw(CLError(status))
                end
            end
        catch err 
            throw(err)
        end
        return Program(program_id)
    end
end


# high level api
function build(p::Program, d::Device)
    @check api.clBuildProgram(p.id, 0, C_NULL, C_NULL, C_NULL, C_NULL)
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
