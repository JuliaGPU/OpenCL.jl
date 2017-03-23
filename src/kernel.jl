# OpenCL.Kernel

type Kernel <: CLObject
    id :: CL_kernel

    function Kernel(k::CL_kernel, retain=false)
        if retain
            @check api.clRetainKernel(k)
        end
        kernel = new(k)
        finalizer(kernel, _finalize)
        return kernel
    end
end

function _finalize(k::Kernel)
    if k.id != C_NULL
        @check api.clReleaseKernel(k.id)
        k.id = C_NULL
    end
end

Base.pointer(k::Kernel) = k.id

Base.show(io::IO, k::Kernel) = begin
    print(io, "OpenCL.Kernel(\"$(k[:name])\" nargs=$(k[:num_args]))")
end

Base.getindex(k::Kernel, kinfo::Symbol) = info(k, kinfo)

function Kernel(p::Program, kernel_name::String)
    for (dev, status) in info(p, :build_status)
        if status != CL_BUILD_SUCCESS
            msg = "OpenCL.Program has to be built before Kernel constructor invoked"
            throw(ArgumentError(msg))
        end
    end
    err_code = Ref{CL_int}()
    kernel_id = api.clCreateKernel(p.id, kernel_name, err_code)
    if err_code[] != CL_SUCCESS
        throw(CLError(err_code[]))
    end
    return Kernel(kernel_id)
end

immutable LocalMem{T}
    nbytes::Csize_t
end

LocalMem{T}(::Type{T}, len::Integer) = begin
    @assert len > 0
    nbytes = sizeof(T) * len
    return LocalMem{T}(convert(Csize_t, nbytes))
end

Base.ndims(l::LocalMem) = 1
Base.eltype{T}(l::LocalMem{T}) = T
Base.sizeof{T}(l::LocalMem{T}) = l.nbytes
Base.length{T}(l::LocalMem{T}) = Int(l.nbytes ÷ sizeof(T))

function set_arg!(k::Kernel, idx::Integer, arg::Void)
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
    arg_boxed = Ref{typeof(arg.id)}(arg.id)
    @check api.clSetKernelArg(k.id, cl_uint(idx-1), sizeof(CL_mem), arg_boxed)
    return k
end

function set_arg!(k::Kernel, idx::Integer, arg::LocalMem)
    @assert idx > 0 "Kernel idx must be bigger 0"
    @check api.clSetKernelArg(k.id, cl_uint(idx-1), arg.nbytes, C_NULL)
    return k
end

#TODO: type safe calling of set args for kernel (with clang)
function set_arg!{T}(k::Kernel, idx::Integer, arg::T)
    @assert idx > 0 "Kernel idx must be bigger 0"
    if !isbits(T) # TODO add more thorough mem layout checks and the clang stuff
        error("Only isbits types allowed. Found: $T")
    end
    boxed_arg = Ref{T}(arg)
    @check api.clSetKernelArg(k.id, cl_uint(idx - 1), sizeof(T), boxed_arg)
    return k
end

function set_args!(k::Kernel, args...)
    for (i, a) in enumerate(args)
        set_arg!(k, i, a)
    end
end

function work_group_info(k::Kernel, winfo::CL_kernel_work_group_info, d::Device)
    if (winfo == CL_KERNEL_LOCAL_MEM_SIZE ||
        winfo == CL_KERNEL_PRIVATE_MEM_SIZE)
        result1 = Ref{CL_ulong}(0)
        @check api.clGetKernelWorkGroupInfo(k.id, d.id, winfo,
                                            sizeof(CL_ulong), result1, C_NULL)
        return Int(result1[])
    elseif winfo == CL_KERNEL_COMPILE_WORK_GROUP_SIZE
        # Intel driver has a bug so we can't query the required size.
        # As specified by [1] the return value in this case is size_t[3].
        # [1] https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clGetKernelWorkGroupInfo.html
        @assert sizeof(Csize_t) == sizeof(Int)
        result2 = Vector{Int}(3)
        @check api.clGetKernelWorkGroupInfo(k.id, d.id, winfo, 3*sizeof(Int), result2, C_NULL)
        return result2
    else
        result = Ref{Csize_t}(0)
        @check api.clGetKernelWorkGroupInfo(k.id, d.id, winfo,
                                            sizeof(CL_ulong), result, C_NULL)
        return Int(result[])
    end
end

function work_group_info(k::Kernel, winfo::Symbol, d::Device)
    if winfo == :size
        work_group_info(k, CL_KERNEL_WORK_GROUP_SIZE, d)
    elseif winfo == :compile_size
        work_group_info(k, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, d)
    elseif winfo == :local_mem_size
        work_group_info(k, CL_KERNEL_LOCAL_MEM_SIZE, d)
    elseif winfo == :private_mem_size
        work_group_info(k, CL_KERNEL_PRIVATE_MEM_SIZE, d)
    elseif winfo == :prefered_size_multiple
        work_group_info(k, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, d)
    else
        throw(ArgumentError(("Unknown work_group_info flag: :$winfo")))
    end
end

# produce a cl.call thunk with kernel queue, global/local sizes
Base.getindex(k::Kernel, args...) = begin
    if length(args) < 2 || length(args) > 3
        throw(ArgumentError("kernel must be called with a queue & global size as arguments"))
    end
    if !(isa(args[1], CmdQueue))
        throw(ArgumentError("kernel first argument must a a CmdQueue"))
    end
    if !(isa(args[2], Dims)) || length(args[2]) > 3
        throw(ArgumentError("kernel global size must be of Dims type (dim <= 3)"))
    end
    if length(args) == 3 && (!(isa(args[3], Dims)) || length(args[3]) > 3)
        throw(ArgumentError("kernel local size must be of Dims type (dim <= 3)"))
    end
    queue = args[1]
    global_size = args[2]
    local_size  = length(args) == 3 ? args[3] : nothing
    # TODO: we cannot pass keywords in anon functions yet, return kernel call thunk
    return (args...) -> queue(k, global_size, local_size, args...)
end

# blocking kernel call that finishes queue
function (q::CmdQueue)(k::Kernel, global_work_size, local_work_size,
                      args...; global_work_offset=nothing,
                      wait_on::Union{Void,Vector{Event}}=nothing)
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
                                wait_on::Union{Void,Vector{Event}}=nothing)
    device = q[:device]
    max_work_dim = device[:max_work_item_dims]
    work_dim     = length(global_work_size)
    if work_dim > max_work_dim
        throw(ArgumentError("global_work_size has max dim of $max_work_dim"))
    end
    gsize = Array{Csize_t}(work_dim)
    for (i, s) in enumerate(global_work_size)
        gsize[i] = s
    end

    goffset = C_NULL
    if global_work_offset !== nothing
        if length(global_work_offset) > max_work_dim
            throw(ArgumentError("global_work_offset has max dim of $max_work_dim"))
        end
        if length(global_work_offset) != work_dim
            throw(ArgumentError("global_work_size and global_work_offset have differing dims"))
        end
        goffset = Array{Csize_t}(work_dim)
        for (i, o) in enumerate(global_work_offset)
            goffset[i] = o
        end
    end

    lsize = C_NULL
    if local_work_size !== nothing
        if length(local_work_size) > max_work_dim
            throw(ArgumentError("local_work_offset has max dim of $max_work_dim"))
        end
        if length(local_work_size) != work_dim
            throw(ArgumentError("global_work_size and local_work_size have differing dims"))
        end
        lsize = Array{Csize_t}(work_dim)
        for (i, s) in enumerate(local_work_size)
            lsize[i] = s
        end
    end

    if wait_on !== nothing
        n_events = cl_uint(length(wait_on))
        wait_event_ids = [evt.id for evt in wait_on]
    else
        n_events = cl_uint(0)
        wait_event_ids = C_NULL
    end

    ret_event = Ref{CL_event}()
    @check api.clEnqueueNDRangeKernel(q.id, k.id, cl_uint(work_dim), goffset, gsize, lsize,
                                      n_events, wait_event_ids, ret_event)
    return Event(ret_event[], retain=false)
end


function enqueue_task(q::CmdQueue, k::Kernel; wait_for=nothing)
    n_evts  = 0
    evt_ids = C_NULL
    #TODO: this should be split out into its own function
    if wait_for !== nothing
        if isa(wait_for, Event)
            n_evts = 1
            evt_ids = [wait_for.id]
        else
            @assert all([isa(evt, Event) for evt in wait_for])
            n_evts = length(wait_for)
            evt_ids = [evt.id for evt in wait_for]
        end
    end
    ret_event = Ref{CL_event}()
    @check api.clEnqueueTask(q.id, k.id, n_evts, evt_ids, ret_event)
    return ret_event[]
end

let name(k::Kernel) = begin
        size = Ref{Csize_t}()
        @check api.clGetKernelInfo(k.id, CL_KERNEL_FUNCTION_NAME,
                                   0, C_NULL, size)
        result = Vector{Cchar}(size[])
        @check api.clGetKernelInfo(k.id, CL_KERNEL_FUNCTION_NAME,
                                   size[], result, size)
        return CLString(result)
    end

    num_args(k::Kernel) = begin
        ret = Ref{CL_uint}()
        @check api.clGetKernelInfo(k.id, CL_KERNEL_NUM_ARGS,
                                   sizeof(CL_uint), ret, C_NULL)
        return ret[]
    end

    reference_count(k::Kernel) = begin
        ret = Ref{CL_uint}()
        @check api.clGetKernelInfo(k.id, CL_KERNEL_REFERENCE_COUNT,
                                   sizeof(CL_uint), ret, C_NULL)
        return ret[]
    end

    program(k::Kernel) = begin
        ret = Ref{CL_program}()
        @check api.clGetKernelInfo(k.id, CL_KERNEL_PROGRAM,
                                   sizeof(CL_program), ret, C_NULL)
        return Program(ret[], retain=true)
    end

    attributes(k::Kernel) = begin
        size = Ref{Csize_t}()
        api.clGetKernelInfo(k.id, CL_KERNEL_ATTRIBUTES,
                            0, C_NULL, size)
        if size[] <= 1
            return ""
        end
        result = Vector{CL_char}(size[])
        @check api.clGetKernelInfo(k.id, CL_KERNEL_ATTRIBUTES,
                                   size[], result, size)
        return CLString(result)
    end

    const info_map = Dict{Symbol, Function}(
        :name => name,
        :num_args => num_args,
        :reference_count => reference_count,
        :program => program,
        :attributes => attributes
    )

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
# OpenCL 1.2 function
#TODO: get_arg_info(k::Kernel, idx, param)
#TODO: enqueue_async_kernel()
