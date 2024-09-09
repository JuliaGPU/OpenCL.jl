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
        clReleaseProgram(p)
        p.id = C_NULL
    end
end

Base.show(io::IO, p::Program) = begin
    ptr_val = convert(UInt, Base.pointer(p))
    ptr_address = "0x$(string(ptr_val, base = 16, pad = Sys.WORD_SIZE>>2))"
    print(io, "OpenCL.Program(@$ptr_address)")
end

Base.unsafe_convert(::Type{cl_program}, p::Program) = p.id

Base.pointer(p::Program) = p.id

function Program(; source=nothing, binaries=nothing, il=nothing)
    if count(!isnothing, (source, binaries, il)) != 1
        throw(ArgumentError("Program must be source, binary, or intermediate language"))
    end
    if source !== nothing
        byte_source = [String(source)]
        err_code = Ref{Cint}()
        program_id = clCreateProgramWithSource(context(), 1, byte_source, C_NULL, err_code)
        if err_code[] != CL_SUCCESS
            throw(CLError(err_code[]))
        end

    elseif il !== nothing
        err_code = Ref{Cint}()
        program_id = clCreateProgramWithIL(context(), il, length(il), err_code)
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
            program_id = clCreateProgramWithBinary(context(), ndevices, device_ids, bin_lengths,
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
    err = unchecked_clBuildProgram(p, cl_uint(ndevices), device_ids, opts, C_NULL, C_NULL)
    for (dev, status) in p.build_status
        if status == CL_BUILD_ERROR
            println(stderr, "Couldn't compile kernel: ")
            source = p.source
            print_with_linenumbers(source, "    ", stderr)
            println(stderr, "With following build error:")
            println(stderr, p.build_log[dev])
            raise && err # throw the build error when raise!
        end
    end
    if err != CL_SUCCESS
       throw(CLError(err))
    end
    return p
end

function Base.getproperty(p::Program, s::Symbol)
    if s == :reference_count
        count = Ref{Cuint}()
        clGetProgramInfo(p, CL_PROGRAM_REFERENCE_COUNT, sizeof(Cuint), count, C_NULL)
        return Int(count[])
    elseif s == :num_devices
        count = Ref{Cuint}()
        clGetProgramInfo(p, CL_PROGRAM_NUM_DEVICES, sizeof(Cuint), count, C_NULL)
        return Int(count[])
    elseif s == :devices
        device_ids = Vector{cl_device_id}(undef, p.num_devices)
        clGetProgramInfo(p, CL_PROGRAM_DEVICES, sizeof(device_ids), device_ids, C_NULL)
        return [Device(id) for id in device_ids]
    elseif s == :source
        src_len = Ref{Csize_t}()
        clGetProgramInfo(p, CL_PROGRAM_SOURCE, 0, C_NULL, src_len)
        src_len[] <= 1 && return nothing
        src = Vector{Cchar}(undef, src_len[])
        clGetProgramInfo(p, CL_PROGRAM_SOURCE, src_len[], src, C_NULL)
        return CLString(src)
    elseif s == :binary_sizes
        sizes = Vector{Csize_t}(undef, p.num_devices)
        clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(sizes), sizes, C_NULL)
        return sizes
    elseif s == :binaries
        sizes = p.binary_sizes

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
        clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(bins), bins, C_NULL)

        binary_dict = Dict{Device, Array{UInt8}}()
        bidx = 1
        for (i, d) in enumerate(p.devices)
            if sizes[i] > 0
                binary_dict[d] = bin_arrays[bidx]
                bidx += 1
            end
        end
        return binary_dict
    elseif s == :context
        ctx = Ref{cl_context}()
        clGetProgramInfo(p, CL_PROGRAM_CONTEXT, sizeof(cl_context), ctx, C_NULL)
        return Context(ctx[], retain=true)
    elseif s == :build_status
        status_dict = Dict{Device, cl_build_status}()
        for device in p.devices
            status = Ref{cl_build_status}()
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), status, C_NULL)
            status_dict[device] = status[]
        end
        return status_dict
    elseif s == :build_log
        log_dict = Dict{Device, String}()
        for device in p.devices
            size = Ref{Csize_t}()
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, 0, C_NULL, size)
            log = Vector{Cchar}(undef, size[])
            clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, size[], log, C_NULL)
            log_dict[device] = CLString(log)
        end
        return log_dict
    else
        return getfield(p, s)
    end
end

#OpenCL 1.2
#TODO: create_program_with_built_in_kernels(ctx, devices, kernel_names)
#TODO: link_program(ctx, programs; options=[], devices=None)
#TODO: unload_platform_compiler(platform)
