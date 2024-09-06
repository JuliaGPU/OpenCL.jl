# OpenCL.Program

using Printf

mutable struct Program <: CLObject
    id::cl_program

    function Program(program_id::cl_program; retain::Bool=false)
        if retain
            clRetainProgram(program_id)
        end
        p = new(program_id)
        finalizer(_finalize, p)
        return p
    end
end

function _finalize(p::Program)
    if p.id != C_NULL
        clReleaseProgram(p.id)
        p.id = C_NULL
    end
end

Base.show(io::IO, p::Program) = begin
    ptr_val = convert(UInt, Base.pointer(p))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Program(@$ptr_address)")
end

Base.pointer(p::Program) = p.id

Base.getindex(p::Program, pinfo::Symbol) = info(p, pinfo)

function Program(ctx::Context; source=nothing, binaries=nothing, il=nothing)
    if count(!isnothing, (source, binaries, il)) != 1
        throw(ArgumentError("Program must be source, binary, or intermediate language"))
    end
    if source !== nothing
        byte_source = [String(source)]
        err_code = Ref{Cint}()
        program_id = clCreateProgramWithSource(ctx.id, 1, byte_source, C_NULL, err_code)
        if err_code[] != CL_SUCCESS
            throw(CLError(err_code[]))
        end

    elseif il !== nothing
        err_code = Ref{Cint}()
        program_id = clCreateProgramWithIL(ctx.id, il, length(il), err_code)
        if err_code[] != CL_SUCCESS
            throw(CLError(err_code[]))
        end

    elseif binaries !== nothing
        ndevices = length(binaries)
        device_ids = Vector{cl_device_id}(undef, ndevices)
        bin_lengths = Vector{Csize_t}(undef, ndevices)
        binary_status = Vector{Cint}(undef, ndevices)
        binary_ptrs= Vector{Ptr{UInt8}}(undef, ndevices)
        try
            for (i, (dev, bin)) in enumerate(binaries)
                device_ids[i] = dev.id
                bin_lengths[i] = length(bin)
                binary_ptrs[i] = Base.unsafe_convert(Ptr{UInt8}, pointer(bin))
            end
            err_code = Ref{Cint}()
            program_id = clCreateProgramWithBinary(ctx.id, ndevices, device_ids, bin_lengths,
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
    end
    Program(program_id)
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
    err = unchecked_clBuildProgram(p.id, cl_uint(ndevices), device_ids, opts, C_NULL, C_NULL)
    for (dev, status) in cl.info(p, :build_status)
        if status == cl.CL_BUILD_ERROR
            println(stderr, "Couldn't compile kernel: ")
            source = info(p, :source)
            print_with_linenumbers(source, "    ", stderr)
            println(stderr, "With following build error:")
            println(stderr, cl.info(p, :build_log)[dev])
            raise && err # throw the build error when raise!
        end
    end
    if err != CL_SUCCESS
       throw(CLError(err))
    end
    return p
end

function info(p::Program, pinfo::Symbol)
    num_devices(p::Program) = begin
        ret = Ref{Cuint}()
        clGetProgramInfo(p.id, CL_PROGRAM_NUM_DEVICES, sizeof(ret), ret, C_NULL)
        return ret[]
    end

    devices(p::Program) = begin
        ndevices = num_devices(p)
        device_ids = Vector{cl_device_id}(undef, ndevices)
        clGetProgramInfo(p.id, CL_PROGRAM_DEVICES,
                             sizeof(cl_device_id) * ndevices, device_ids, C_NULL)
        return [Device(device_ids[i]) for i in 1:ndevices]
    end

    build_status(p::Program) = begin
        status_dict = Dict{Device, cl_build_status}()
        status = Ref{cl_build_status}()
        for d in devices(p)
            clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_STATUS,
                                      sizeof(cl_build_status), status, C_NULL)
            status_dict[d] = status[]
        end
        return status_dict
    end

    build_logs(p::Program) = begin
        logs = Dict{Device, String}()
        for d in devices(p)
            log_len = Ref{Csize_t}()
            clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_LOG,
                                      0, C_NULL, log_len)
            if log_len[] == 0
                logs[d] = ""
                continue
            end
            log_bytestring = Vector{Cchar}(undef, log_len[])
            clGetProgramBuildInfo(p.id, d.id, CL_PROGRAM_BUILD_LOG,
                                      log_len[], log_bytestring, C_NULL)
            logs[d] = CLString(log_bytestring)
        end
        return logs
    end

    binaries(p::Program) = begin
        binary_dict = Dict{Device, Array{UInt8}}()
        slen = Ref{Csize_t}()
        clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES,
                                    0, C_NULL, slen)

        sizes = zeros(Csize_t, slen[])
        clGetProgramInfo(p.id, CL_PROGRAM_BINARY_SIZES,
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
        clGetProgramInfo(p.id, CL_PROGRAM_BINARIES,
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
        src_len = Ref{Csize_t}()
        clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, 0, C_NULL, src_len)
        src_len[] <= 1 && return nothing
        src = Vector{Cchar}(undef, src_len[])
        clGetProgramInfo(p.id, CL_PROGRAM_SOURCE, src_len[], src, C_NULL)
        return CLString(src)
    end

    context(p::Program) = begin
        ret = Ref{cl_context}()
        clGetProgramInfo(p.id, CL_PROGRAM_CONTEXT,
                                    sizeof(cl_context), ret, C_NULL)
        return Context(ret[], retain = true)
    end

    reference_count(p::Program) = begin
        ret = Ref{Cuint}()
        clGetProgramInfo(p.id, CL_PROGRAM_REFERENCE_COUNT,
                                    sizeof(Cuint), ret, C_NULL)
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
