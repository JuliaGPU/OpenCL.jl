# OpenCL.Program

type Program <: CLObject
    id::CL_program
    binary::Bool

    function Program(program_id::CL_program;
                     retain::Bool=false, binary::Bool=false)
        if retain
            @check api.clRetainProgram(program_id)
        end
        p = new(program_id, binary)
        finalizer(p, prog -> release!(prog))
        return p
    end
end

Base.show(io::IO, p::Program) = begin
    ptr_address = "0x$(hex(unsigned(Base.pointer(p)), WORD_SIZE>>2))"
    print(io, "OpenCL.Program(@$ptr_address)")
end

Base.pointer(p::Program) = p.id

Base.getindex(p::Program, pinfo::Symbol) = info(p, pinfo)

function release!(p::Program)
    if p.id != C_NULL
        @check api.clReleaseProgram(p.id)
        p.id = C_NULL
    end
end

function Program(ctx::Context; source=nothing, binaries=nothing)
    local program_id::CL_program
    if source != nothing && binaries != nothing
        throw(ArgumentError("Program be source or binary"))
    end
    if source != nothing
        byte_source = [bytestring(source)]
        err_code = Array(CL_int, 1)
        program_id = api.clCreateProgramWithSource(ctx.id, 1, byte_source, C_NULL, err_code)
        if err_code[1] != CL_SUCCESS
            throw(CLError(err_code[1]))
        end
        return Program(program_id, binary=false)

    elseif binaries != nothing
        ndevices = length(binaries)
        device_ids = Array(CL_device_id, ndevices)
        bin_lengths = Array(Csize_t, ndevices)
        binary_status = Array(CL_int, ndevices)
        binary_ptrs= Array(Ptr{UInt8}, ndevices)
        try
            for (i, (dev, bin)) in enumerate(binaries)
                device_ids[i] = dev.id
                bin_lengths[i] = length(bin)
                binary_ptrs[i] = Compat.unsafe_convert(Ptr{UInt8}, pointer(bin))
            end
            err_code = Array(CL_int, 1)
            program_id = api.clCreateProgramWithBinary(ctx.id, ndevices, device_ids, bin_lengths,
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
        return Program(program_id, binary=true)
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

let
    num_devices(p::Program) = begin
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
        status_dict = Dict{Device, CL_build_status}()
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
        logs = Dict{Device, ASCIIString}()
        log_len = Csize_t[0]
        log_bytestring = Array(Cchar, log_len[1])
        for d in devices(p)
            @check api.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_LOG,
                                            0, C_NULL, log_len)
            if log_len[1] == 0
                logs[d] = ""
                continue
            end
            resize!(log_bytestring, log_len[1])
            @check api.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_LOG,
                                            log_len[1], log_bytestring, C_NULL)
            logs[d] = bytestring(pointer(log_bytestring))
        end
        return logs
    end

    binaries(p::Program) = begin
        binary_dict = Dict{Device, Array{UInt8}}()
        slen = Array(CL_int, 1)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES,
                                    0, C_NULL, pointer(slen))

        sizes = zeros(Csize_t, slen[1])
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES,
                                    slen[1], sizes, C_NULL)
        bins = Array(Ptr{UInt8}, length(sizes))
        # keep a reference to the underlying binary arrays
        # as storing the pointer to the array hides the additional
        # reference from julia's garbage collector
        bin_arrays = Any[]
        for (i, s) in enumerate(sizes)
            if s > 0
                bin = Array(UInt8, s)
                bins[i] = pointer(bin)
                push!(bin_arrays, bin)
            else
                bins[i] = Compat.unsafe_convert(Ptr{UInt8}, C_NULL)
            end
        end
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARIES,
                                    length(sizes) * sizeof(Ptr{UInt8}),
                                    bins, C_NULL)
        bidx = 1
        for (i, d) in enumerate(devices(p))
            if sizes[i] > 0
                binary_dict[d] = bin_arrays[bidx]
                bidx += 1
            end
        end
        return binary_dict
    end

    source(p::Program) = begin
        p.binary && throw(CLError(-45))
        src_len = Array(Csize_t, 1)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, 0, C_NULL, src_len)
        src_len[1] <= 1 && return nothing
        src = Array(Cchar, src_len[1])
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, src_len[1], src, C_NULL)
        return bytestring(pointer(src))
    end

    context(p::Program) = begin
        ret = Array(CL_context, 1)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_CONTEXT,
                                    sizeof(CL_context), ret, C_NULL)
        return Context(ret[1], retain = true)
    end

    reference_count(p::Program) = begin
        ret = Array(CL_uint, 1)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_REFERENCE_COUNT,
                                    sizeof(CL_uint), ret, C_NULL)
        return ret[1]
    end

    const info_map = @compat Dict{Symbol, Function}(
        :reference_count => reference_count,
        :devices => devices,
        :context => context,
        :num_devices => num_devices,
        :source => source,
        :binaries => binaries,
        :build_log => build_logs,
        :build_status => build_status,
    )

    function info(p::Program, pinfo::Symbol)
        try
            func = info_map[pinfo]
            func(p)
        catch err
            if isa(err, KeyError)
                throw(ArgumentError("OpenCL.Program has no info for $pinfo"))
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
