# OpenCL.Program

using Printf

mutable struct Program <: CLObject
    id::CL_program
    binary::Bool

    function Program(program_id::CL_program;
                     retain::Bool=false, binary::Bool=false)
        if retain
            @check api.clRetainProgram(program_id)
        end
        p = new(program_id, binary)
        finalizer(_finalize, p)
        return p
    end
end

function _finalize(p::Program)
    if p.id != C_NULL
        @check api.clReleaseProgram(p.id)
        p.id = C_NULL
    end
end

Base.show(io::IO, p::Program) = begin
    ptr_val = convert(UInt, Base.pointer(p))
    ptr_address = "0x$(string(ptr_val, base=16))"
    print(io, "OpenCL.Program(@$ptr_address)")
end

Base.pointer(p::Program) = p.id

Base.getindex(p::Program, pinfo::Symbol) = info(p, pinfo)

function Program(ctx::Context; source=nothing, binaries=nothing)
    local program_id::CL_program
    if source !== nothing && binaries !== nothing
        throw(ArgumentError("Program be source or binary"))
    end
    if source !== nothing
        byte_source = [String(source)]
        err_code = Ref{CL_int}()
        program_id = api.clCreateProgramWithSource(ctx.id, 1, byte_source, C_NULL, err_code)
        if err_code[] != CL_SUCCESS
            throw(CLError(err_code[]))
        end
        return Program(program_id, binary=false)

    elseif binaries !== nothing
        ndevices = length(binaries)
        device_ids = Vector{CL_device_id}(undef, ndevices)
        bin_lengths = Vector{Csize_t}(undef, ndevices)
        binary_status = Vector{CL_int}(undef, ndevices)
        binary_ptrs= Vector{Ptr{UInt8}}(undef, ndevices)
        try
            for (i, (dev, bin)) in enumerate(binaries)
                device_ids[i] = dev.id
                bin_lengths[i] = length(bin)
                binary_ptrs[i] = Base.unsafe_convert(Ptr{UInt8}, pointer(bin))
            end
            err_code = Ref{CL_int}()
            program_id = api.clCreateProgramWithBinary(ctx.id, ndevices, device_ids, bin_lengths,
                                                       binary_ptrs, binary_status, err_code)
            if err_code[] != CL_SUCCESS
                throw(CLError(err_code[]))
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

function print_with_linenumbers(text, pad = "", io = STDOUT)
    for (i,line) in enumerate(split(text, "\n"))
        println(io, @sprintf("%s%-4d: %s", pad, i, line))
    end
end

#TODO: build callback...
function build!(p::Program; options = "", raise = true)
    opts = String(options)
    ndevices = 0
    device_ids = C_NULL
    err = api.clBuildProgram(p.id, cl_uint(ndevices), device_ids, opts, C_NULL, C_NULL)
    if err != CL_BUILD_PROGRAM_FAILURE
       @check err
    end
    for (dev, status) in cl.info(p, :build_status)
        if status == cl.CL_BUILD_ERROR
            println(STDERR, "Couldn't compile kernel: ")
            source = info(p, :source)
            print_with_linenumbers(source, "    ", STDERR)
            println(STDERR, "With following build error:")
            println(STDERR, cl.info(p, :build_log)[dev])
            raise && @check err # throw the build error when raise!
        end
    end
    return p
end

function info(p::Program, pinfo::Symbol)
    num_devices(p::Program) = begin
        ret = Ref{CL_uint}()
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_NUM_DEVICES, sizeof(ret), ret, C_NULL)
        return ret[]
    end

    devices(p::Program) = begin
        ndevices = num_devices(p)
        device_ids = Vector{CL_device_id}(undef, ndevices)
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_DEVICES,
                                    sizeof(CL_device_id) * ndevices, device_ids, C_NULL)
        return [Device(device_ids[i]) for i in 1:ndevices]
    end

    build_status(p::Program) = begin
        status_dict = Dict{Device, CL_build_status}()
        status = Ref{CL_build_status}()
        for d in devices(p)
            @check api.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_STATUS,
                                             sizeof(CL_build_status), status, C_NULL)
            status_dict[d] = status[]
        end
        return status_dict
    end

    build_logs(p::Program) = begin
        logs = Dict{Device, String}()
        for d in devices(p)
            log_len = Ref{Csize_t}()
            @check api.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_LOG,
                                             0, C_NULL, log_len)
            if log_len[] == 0
                logs[d] = ""
                continue
            end
            log_bytestring = Vector{CL_char}(undef, log_len[])
            @check api.clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_LOG,
                                            log_len[], log_bytestring, C_NULL)
            logs[d] = CLString(log_bytestring)
        end
        return logs
    end

    binaries(p::Program) = begin
        binary_dict = Dict{Device, Array{UInt8}}()
        slen = Ref{Csize_t}()
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES,
                                    0, C_NULL, slen)

        sizes = zeros(Csize_t, slen[])
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES,
                                    slen[], sizes, C_NULL)
        bins = Vector{Ptr{UInt8}}(undef, length(sizes))
        # keep a reference to the underlying binary arrays
        # as storing the pointer to the array hides the additional
        # reference from julia's garbage collector
        bin_arrays = Any[]
        for (i, s) in enumerate(sizes)
            if s > 0
                bin = Vector{UInt8}(undef, s)
                bins[i] = pointer(bin)
                push!(bin_arrays, bin)
            else
                bins[i] = Base.unsafe_convert(Ptr{UInt8}, C_NULL)
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
        src_len = Ref{Csize_t}()
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, 0, C_NULL, src_len)
        src_len[] <= 1 && return nothing
        src = Vector{Cchar}(undef, src_len[])
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, src_len[], src, C_NULL)
        return CLString(src)
    end

    context(p::Program) = begin
        ret = Ref{CL_context}()
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_CONTEXT,
                                    sizeof(CL_context), ret, C_NULL)
        return Context(ret[], retain = true)
    end

    reference_count(p::Program) = begin
        ret = Ref{CL_uint}()
        @check api.clGetProgramInfo(p.id, CL_PROGRAM_REFERENCE_COUNT,
                                    sizeof(CL_uint), ret, C_NULL)
        return ret[]
    end

    info_map = Dict{Symbol, Function}(
        :reference_count => reference_count,
        :devices => devices,
        :context => context,
        :num_devices => num_devices,
        :source => source,
        :binaries => binaries,
        :build_log => build_logs,
        :build_status => build_status,
    )

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

#OpenCL 1.2
#TODO: create_program_with_built_in_kernels(ctx, devices, kernel_names)
#TODO: link_program(ctx, programs; options=[], devices=None)
#TODO: unload_platform_compiler(platform)
