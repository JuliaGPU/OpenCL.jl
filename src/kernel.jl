
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
    print(io, "<OpenCL.Kernel :$(k[:name]) nargs=$(k[:num_args])>")
end 

Base.getindex(k::Kernel, kinfo::Symbol) = info(k, kinfo)

function release!(k::Kernel)
    if k.id != C_NULL
        @check api.clReleaseKernel(k.id)
        k.id = C_NULL
    end
end

function Kernel(p::Program, kernel_name::String)
    for (dev, status) in info(p, :build_status)
        if status != CL_BUILD_SUCCESS
            msg = "OpenCL.Program has to be built before Kernel constructor invoked"
            throw(ArgumentError(msg))
        end
    end
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

LocalMemory(x::Integer) = begin
    @assert x > 0
    return LocalMemory(convert(Csize_t, x))
end

function set_arg!(k::Kernel, idx::Integer, arg::Nothing)
    @assert idx > 0
    @check api.clSetKernelArg(k.id, cl_uint(idx-1), sizeof(CL_mem), C_NULL)
    return k
end

function set_arg!(k::Kernel, idx::Integer, arg::Ptr{Void})
    if arg != C_NULL
        throw(AttributeError("set_arg! for void pointer $arg is undefined"))
    end
    set_arg!(k, idx, nothing)
end

function set_arg!(k::Kernel, idx::Integer, arg::CLMemObject)
    @assert idx > 0
    @check api.clSetKernelArg(k.id, cl_uint(idx-1), sizeof(CL_mem), [arg.id])
    return k
end

function set_arg!(k::Kernel, idx::Integer, arg::LocalMemory)
    @assert idx > 0
    @check api.clSetKernelArg(k.id, cl_uint(idx-1), arg.size, C_NULL)
    return k
end

#TODO: vector types...
#TODO: type safe calling of set args for kernel

# set scalar/vector kernel args
for cl_type in [:CL_char, :CL_uchar, :CL_short, :CL_ushort,
                :CL_int,  :CL_uint,  :CL_long,  :CL_ulong,
                :CL_half, :CL_float, :CL_double]
    @eval begin
        function set_arg!(k::Kernel, idx::Integer, arg::$cl_type)
            @assert idx > 0
            boxed_arg = $cl_type[arg,]
            @check api.clSetKernelArg(k.id, cl_uint(idx-1),
                                      sizeof($cl_type), boxed_arg)
            return k
        end
    end
end

function set_args!(k::Kernel, args...)
    for (i, a) in enumerate(args)
        set_arg!(k, i, a)
    end
end


function private_mem_size(k::Kernel, d::Device)
    ret = Csize_t[0]
    @check api.clGetKernelWorkGroupInfo(k.id, d.id, 
                                        CL_KERNEL_PRIVATE_MEM_SIZE,
                                        sizeof(Csize_t), ret, C_NULL)
    return int(ret[1])
end

function local_mem_size(k::Kernel, d::Device)
    ret = Csize_t[0]
    @check api.clGetKernelWorkGroupInfo(k.id, d.id, 
                                        CL_KERNEL_PRIVATE_MEM_SIZE,
                                        sizeof(Csize_t), ret, C_NULL)
    return int(ret[1])
end

function work_group_size(k::Kernel, d::Device)
    ret = Csize_t[0, 0, 0]
    @check api.clGetKernelWorkGroupInfo(k.id, d.id, 
                                        CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 
                                        sizeof(ret), ret, C_NULL)
    return int(ret)
end

# blocking kernel call that finishes queue
function call(q::CmdQueue, k::Kernel, global_work_size, local_work_size, args...;
              global_work_offset=nothing,
              wait_on::Union(Nothing, Vector{Event})=nothing)
    set_args!(k, args...)
    evt = enqueue_kernel(q, k, 
                         global_work_size,
                         local_work_size,
                         global_work_offset=global_work_offset,
                         wait_on=wait_on)
    finish(q)
    return evt
end

function enqueue_kernel(q::CmdQueue, k::Kernel, global_work_size)
    enqueue_kernel(q, k, global_work_size, nothing)
end

function enqueue_kernel(q::CmdQueue,
                        k::Kernel,
                        global_work_size,
                        local_work_size;
                        global_work_offset=nothing,
                        wait_on::Union(Nothing,Vector{Event})=nothing)
    #TODO: check global work size against max possible global work size
    work_dim = length(global_work_size)
    if work_dim > 3
        throw(AttributeError("global_work_size has max dim of 3"))
    end
    gsize = Array(Csize_t, work_dim)
    for (i, s) in enumerate(global_work_size)
        gsize[i] = s
    end

    goffset = C_NULL 
    if global_work_offset != nothing 
        if length(global_work_offset) > 3
            throw(AttributeError("global_work_offset has max dim of 3"))
        end
        if length(global_work_offset) != work_dim 
            throw(AttributeError("global_work_offset dim must match global_work_size dim"))
        end
        goffset = Array(Csize_t, work_dim)
        for (i, o) in enumerate(global_work_offset)
            goffset[i] = o
        end
    end

    lsize = C_NULL
    if local_work_size != nothing
        #TODO: check local work size against max possible local work size....
        if length(local_work_size) > 3
            throw(AttributeError("local_work_offset has max dim of 3"))
        end
        if length(local_work_size) != work_dim
            throw(AttributeError("global/local work sizes have differing dimensions"))
        end
        lsize = Array(Csize_t, work_dim)
        for (i, s) in enumerate(local_work_size)
            lsize[i] = s
        end
    end

    if wait_on != nothing
        n_events = cl_uint(length(wait_on))
        wait_event_ids = [evt.id for evt in wait_on]
    else
        n_events = cl_uint(0)
        wait_event_ids = C_NULL
    end

    ret_event = Array(CL_event, 1)
    #TODO: Support offsets??? hardcoded to NULL for the time being...
    @check api.clEnqueueNDRangeKernel(q.id, k.id, cl_uint(work_dim), C_NULL, gsize, lsize,
                                      n_events, wait_event_ids, ret_event)
    return Event(ret_event[1], retain=false)
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
#TODO: get_arg_info(k::Kernel, idx, param)
#TODO: ?macro capture_call
#TODO: enqueue_task(q::Queue, k::Kernel; wait_for=None)
#TODO: enqueue_async_kernel()
