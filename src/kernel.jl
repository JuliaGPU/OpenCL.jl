
type Kernel
    id :: CL_kernel

    function Kernel(k::CL_kernel, retain=false)
        if retain
            @check api.clRetainKernel(k)
        end
        kernel = new(k)
        finalizer(kernel, x -> release!(x))
        return kernel
    end
end

Base.pointer(k::Kernel) = k.id
@ocl_object_equality(Kernel)

Base.show(io::IO, k::Kernel) = begin
    print(io, k)
end 

Base.getindex(k::Kernel, kinfo::Symbol) = info(k, kinfo)

function release!(k::Kernel)
    if k.id != C_NULL
        @check api.clReleaseKernel(k.id)
        k.id = C_NULL
    end
end

function Kernel(p::Program, kernel_name::String)
    err_code = Array(CL_int, 1)
    kernel_id = api.clCreateKernel(p.id, kernel_name, err_code)
    if err_code[1] != CL_SUCCESS
        throw(CLError(err_code[1]))
    end
    return Kernel(kernel_id, true)
end

type KernelArg
end

immutable LocalMemory
    size::Csize_t
end

function set_arg!(k::Kernel, idx::Integer, arg::Nothing)
    @assert idx > 0
    @check api.clSetKernelArg(k.id, idx, sizeof(CL_mem), C_NULL)
end

function set_arg!(k::Kernel, idx::Integer, arg::CLMemObject)
    @assert idx > 0
    @check api.clSetKernelArg(k.id, idx, arg.size, arg.id)
end

function set_arg!(k::Kernel, idx::Integer, arg::LocalMemory)
    @assert idx > 0
    @check api.clSetKernelArg(k.id, idx, loc.size, C_NULL)
end

function set_arg!(k::Kernel, idx::Integer, arg::Buffer)
    @assert idx > 0
    @check api.clSetKernelArg(k.id, idx, arg.size, arg.id)
end

#TODO: replace with macros...
let name(k::Kernel) = begin
        size = Array(Csize_t, 1)
        @check api.clGetKernelInfo(k.id, CL_KERNEL_FUNCTION_NAME,
                                   0, C_NULL, size)
        result = Array(Cchar, size[1])
        @check api.clGetKernelInfo(k.id, CL_KERNEL_FUNCTION_NAME, 
                                   size[1], result, size)
        return bytestring(convert(Ptr{Cchar}, result))
    end

    num_args(k::Kernel) = begin
        ret = Array(CL_uint, 1)
        @check api.clGetKernelInfo(k.id, CL_KERNEL_NUM_ARGS,
                                   sizeof(CL_uint), ret, C_NULL)
        return ret[1]
    end

    reference_count(k::Kernel) = begin
        ret = Array(CL_uint, 1)
        @check api.clGetKernelInfo(k.id, CL_KERNEL_REFERENCE_COUNT,
                                   sizeof(CL_uint), ret, C_NULL)
        return ret[1]
    end 

    program(k::Kernel) = begin
        ret = Array(CL_program, 1)
        @check api.clGetKernelInfo(k.id, CL_KERNEL_PROGRAM,
                                   sizeof(CL_program), ret, C_NULL) 
        return Program(ret[1], retain=true)
    end

    attributes(k::Kernel) = begin
        size = Csize_t[0,]
        api.clGetKernelInfo(k.id, CL_KERNEL_ATTRIBUTES, 
                            0, C_NULL, size)
        if size[1] <= 1
            return ""
        end
        result = Array(Cchar, size[1])
        @check api.clGetKernelInfo(k.id, CL_KERNEL_ATTRIBUTES,
                                   size[1], result, size)
        return bytestring(convert(Ptr{Cchar}, result))
    end

    const info_map = (Symbol => Function)[
        :name => name, 
        :num_args => num_args,
        :reference_count => reference_count,
        :program => program,
        :attributes => attributes
    ]

    function info(k::Kernel, kinfo::Symbol)
        try
            func = info_map[kinfo]
            func(k)
        catch err
            if isa(err, KeyError)
                error("OpenCL.Kernel has no info for: $kinfo")
            else
                throw(err)
            end
        end
    end
end

#TODO set_arg sampler...
    
#TODO: local_memory

#TODO: get_info(k::Kernel, param)
#TODO: get_work_group_info(k::Kernel, param, d::Device)
#TODO: get_arg_info(k::Kernel, idx, param)
#TODO: set_arg(k::Kernel, idx, arg)
#TODO: set_args(k::Kernel, args...)
#TODO: call(queue, global_size, local_size, *args, global_offset=None, wait_for=None)
#TODO: ?macro capture_call

#TODO: enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size,
#                              global_work_offset=None, wait_for=None)

#TODO: enqueue_task(q::Queue, k::Kernel; wait_for=None)
#TODO: enqueue_async_kernel()
