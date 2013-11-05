
type Program
    id::CL_program

    function Program(program_id::CL_program; retain=false)
        if retain
            @check api.clRetainProgram(program_id)
        end 
        p = new(program_id)
        finalizer(p, prog -> release!(prog))
        return p
    end
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

#TODO: build callback...
function build!(p::Program; options="", raise=true)
    opts = bytestring(options)
    ndevices = 0
    device_ids = C_NULL
    @check api.clBuildProgram(p.id, cl_uint(ndevices), device_ids, opts, C_NULL, C_NULL)
    if raise
        for (dev, status) in build_status(p)
            if status == CL_BUILD_ERROR
                #TODO: throw(CLBuildError(self.logs[dev], self.logs)
                error("$p build error on device $dev")
            end
        end 
    end
    return p
end

#TODO: return keywords
function build_status(p::Program)
    status_dict = (Device => CL_build_status)[]
    err_code = Array(CL_int, 1)
    status = Array(CL_build_status, 1) 
    for d in devices(p)
        @check api.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_STATUS,
                                         sizeof(CL_build_status), status, C_NULL)
        status_dict[d] = status[1]
    end
    return status_dict
end

function build_logs(p::Program)
    logs = (Device => ASCIIString)[]
    log_len = Array(Csize_t, 1)
    for d in devices(p)
        @check cl.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_LOG, 
                                        0, C_NULL, log_len)
        if log_len == 0
            logs[d] = ""
            continue
        end 
        log_bytestring = Array(Cchar, log_len)
        @check cl.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_LOG,
                                        log_len, log_bytestring, C_NULL)
        logs[d] = bytestring(convert(Ptr{Cchar}, log_bytestring))
    end
    return logs
end 

binaries(p::Program) = begin
    binary_dict = (Device => Array{Cchar})[]
    
    slen = Array(CL_int, 1)
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES, 0, C_NULL, slen)
    sizes = zeros(Csize_t, slen[1])
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES, slen[1], sizes, C_NULL)
    
    size = Array(Csize_t, 1) 
    total_size = reduce(+, 0, sizes)
    bins = Array(Cchar, total_size)
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARIES, 0, C_NULL, size)
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARIES, size[1], bins, C_NULL)
    
    for (i, d) in enumerate(devices(p))
        binary_dict[d] = bins[i]
    end
    return binary_dict
end 

source_code(p::Program) = begin
    src = C_NULL
    src_len = Array(Csize_t, 1)
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, 0, C_NULL, src_len)
    if src_len[1] <= 1
        return nothing
    end 
    src = Array(Cchar, src_len[1])
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, src_len[1], src, C_NULL)
    return bytestring(convert(Ptr{Cchar}, src))
end

#TODO: info property api
num_devices(p::Program) = begin
    ret = Array(CL_uint, 1)
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_NUM_DEVICES, sizeof(ret), ret, C_NULL)
    return ret[1]
end

devices(p::Program) = begin
    err_code = Array(CL_int, 1)
    ndevices = num_devices(p)
    device_ids = Array(CL_device_id, ndevices)
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_DEVICES, 
                                sizeof(CL_device_id) * ndevices, device_ids, C_NULL)
    return [Device(device_ids[i]) for i in 1:ndevices]
end

context(p::Program) = begin
    ret = Array(CL_context, 1)
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_CONTEXT,
                                sizeof(CL_context), ret, C_NULL)
    return Context(ret[1])
end

reference_count(p::Program) = begin
    ret = Array(CL_uint, 1) 
    @check api.clGetProgramInfo(p.id, CL_PROGRAM_REFERENCE_COUNT,
                                sizeof(CL_uint), ret, C_NULL)
    return ret[1]
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
