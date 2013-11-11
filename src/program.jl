
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

Base.pointer(p::Program) = p.id
@ocl_object_equality(Program)

function Base.show(io::IO, p::Program)
    print(io, p)
end

Base.getindex(p::Program, pinfo::Symbol) = info(p, pinfo)

function release!(p::Program)
    if p.id != C_NULL
        @check api.clReleaseProgram(p.id)
        p.id = C_NULL
    end
end

function Program(ctx::Context; source=nothing, binaries=nothing)
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
        lens = Array(Csize_t, n_devices)
        binary_status = Array(CL_int, n_devices)
        binary_ptrs= Array(Ptr{Uint8}, n_devices)
        try
            for (i, (dev, bin)) in enumerate(binaries)
                device_ids[i] = dev.id
                bin_lengths[i] = length(bin)
                binary_ptrs[i] = convert(Ptr{Uint8}, bin)
            end
            err_code = Array(CL_int, 1)
            program_id = api.clCreateProgramWithBinary(ctx.id, n_devices, device_ids, bin_lengths,
                                                       binary_ptrs, binary_status, err_code)
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
        for (dev, status) in info(p, :build_status)
            if status == CL_BUILD_ERROR
                #TODO: throw(CLBuildError(self.logs[dev], self.logs)
                error("$p build error on device $dev")
            end
        end 
    end
    return p
end

#TODO: split out devices into toplevel function...
let num_devices(p::Program) = begin
        ret = Array(CL_uint, 1)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_NUM_DEVICES, sizeof(ret), ret, C_NULL)
        return ret[1]
    end

    devices(p::Program) = begin
        ndevices = num_devices(p)
        device_ids = Array(CL_device_id, ndevices)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_DEVICES, 
                                    sizeof(CL_device_id) * ndevices, device_ids, C_NULL)
        return [Device(device_ids[i]) for i in 1:ndevices]
    end
    
    build_status(p::Program) = begin
        status_dict = (Device => CL_build_status)[]
        
        status = Array(CL_build_status, 1) 
        err_code = Array(CL_int, 1)
        for d in devices(p)
            @check api.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_STATUS,
                                             sizeof(CL_build_status), status, C_NULL)
            status_dict[d] = status[1]
        end
        return status_dict
    end

    build_logs(p::Program) = begin
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
        binary_dict = (Device => Array{Uint8})[]
        
        slen = Array(CL_int, 1)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES, 0, C_NULL, slen)
        sizes = zeros(Csize_t, slen[1])
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES, slen[1], sizes, C_NULL)

        bins = Array(Ptr{Uint8}, length(sizes))
        for (i, s) in enumerate(sizes)
            if s > 0
                bins[i] = convert(Ptr{Uint8}, Array(Uint8, s))
            else
                bins[i] = convert(Ptr{Uint8}, C_NULL)
            end
        end
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARIES,
                                    length(sizes) * sizeof(Ptr{Uint8}), bins, C_NULL)
        for (i, d) in enumerate(devices(p))
            if sizes[i] > 0
                binary_dict[d] = Base.copy(pointer_to_array(bins[i], int(sizes[i])))
            end
        end
        return binary_dict
    end 

    source(p::Program) = begin
        src_len = Array(Csize_t, 1)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, 0, C_NULL, src_len)
        if src_len[1] <= 1
            return nothing
        end
        src = Array(Cchar, src_len[1])
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, src_len[1], src, C_NULL)
        return bytestring(convert(Ptr{Cchar}, src))
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
        
    const info_map = (Symbol => Function)[
        :reference_count => reference_count,
        :devices => devices,
        :context => context,
        :num_devices => num_devices, 
        :source => source, 
        :binaries => binaries,
        :build_log => build_logs,
        :build_status => build_status,
    ]

    function info(p::Program, pinfo::Symbol)
        try 
            func = info_map[pinfo]
            func(p)
        catch err
            if isa(err, KeyError)
                error("OpenCL.Program has no info for $pinfo")
            else
                throw(err)
            end
        end
    end
end

#OpenCL 1.2
#TODO: create_program_with_built_in_kernels(ctx, devices, kernel_names)
#TODO: link_program(ctx, programs; options=[], devices=None)
#TODO: unload_platform_compiler(platform)
