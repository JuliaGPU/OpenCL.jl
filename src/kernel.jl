# OpenCL.Kernel
mutable struct Kernel <: CLObject
    id :: CL_kernel

    function Kernel(k::CL_kernel, retain=false)
        if retain
            @check api.clRetainKernel(k)
        end
        kernel = new(k)
        finalizer(_finalize, kernel)
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

struct LocalMem{T}
    nbytes::Csize_t
end

function LocalMem(::Type{T}, len::Integer) where T
    @assert len > 0
    nbytes = sizeof(T) * len
    return LocalMem{T}(convert(Csize_t, nbytes))
end

Base.ndims(l::LocalMem) = 1
Base.eltype(l::LocalMem{T}) where {T} = T
Base.sizeof(l::LocalMem{T}) where {T} = l.nbytes
Base.length(l::LocalMem{T}) where {T} = Int(l.nbytes ÷ sizeof(T))

function set_arg!(k::Kernel, idx::Integer, arg::Nothing)
    @assert idx > 0
    @check api.clSetKernelArg(k.id, cl_uint(idx-1), sizeof(CL_mem), C_NULL)
    return k
end

function set_arg!(k::Kernel, idx::Integer, arg::Ptr{Nothing})
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

function _contains_different_layout(::Type{T}) where T
    sizeof(T) == 0 && return true
    nfields(T) == 0 && return false
    for fname in fieldnames(T)
        contains_different_layout(fieldtype(T, fname)) && return true
    end
    return false
end

contains_different_layout(::Type{NTuple{3, T}}) where {T <: Union{Float32, Float64, Int8, Int32,
                                                                  Int64, UInt8, UInt32, UInt64}} = true

"""
    contains_different_layout(T)

Empty types and NTuple{3, CLNumber} have different layouts and need to be replaced
(Where `CLNumber <: Union{Float32, Float64, Int8, Int32, Int64, UInt8, UInt32, UInt64}`)
TODO: Float16 + Int16 should also be in CLNumbers
"""
@generated function contains_different_layout(::Type{T}) where T
    :($(_contains_different_layout(T)))
end

function struct2tuple(x::T) where T
    ntuple(nfields(x)) do i
        getfield(x, i)
    end
end

"""
    replace_different_layout(x::T) where T

Replaces types with a layout different from OpenCL.
See [contains_different_layout(T)](@ref) for information what types those are!
"""
function replace_different_layout(x::T) where T
    !contains_different_layout(T) && return x
    if sizeof(x) === 0
        return Int32(0) # zero size not possible in opencl
    elseif nfields(x) == 0
        replace_different_layout((), (x,))
    elseif T <: Tuple
        replace_different_layout((), x)
    else
        replace_different_layout((), struct2tuple(x))
    end
end

replace_different_layout(red::NTuple{N, Any}, rest::Tuple{}) where N = red
function replace_different_layout(red::NTuple{N, Any}, rest) where N
    elem1 = first(rest)
    T = typeof(elem1)
    repl = if sizeof(T) == 0 && nfields(elem1) == 0
        Int32(0)
    elseif contains_different_layout(T)
        replace_different_layout(elem1)
    else
        elem1
    end
    replace_different_layout((red..., repl), Base.tail(rest))
end

# TODO UInt16/Float16?
# Handle different sizes of OpenCL Vec3, which doesn't agree with julia
function replace_different_layout(arg::NTuple{3, T}) where T <: Union{Float32, Float64, Int8, Int32, Int64, UInt8, UInt32, UInt64}
    pad = T(0)
    (arg..., pad)
end

function to_cl_ref(arg::T) where T
    if !Base.datatype_pointerfree(T)
        error("Types should not contain pointers: $T")
    end
    if contains_different_layout(T)
        x = replace_different_layout(arg)
        return Base.RefValue(x), sizeof(x)
    end
    Base.RefValue(arg), sizeof(arg)
end


Base.@pure datatype_align(x::T) where {T} = datatype_align(T)
Base.@pure function datatype_align(::Type{T}) where {T}
    # typedef struct {
    #     uint32_t nfields;
    #     uint32_t alignment : 9;
    #     uint32_t haspadding : 1;
    #     uint32_t npointers : 20;
    #     uint32_t fielddesc_type : 2;
    # } jl_datatype_layout_t;
    field = T.layout + sizeof(UInt32)
    unsafe_load(convert(Ptr{UInt16}, field)) & convert(Int16, 2^9-1)
end


function set_arg!(k::Kernel, idx::Integer, arg::T) where T
    @assert idx > 0 "Kernel idx must be bigger 0"
    ref, tsize = to_cl_ref(arg)
    err = api.clSetKernelArg(k.id, cl_uint(idx - 1), tsize, ref)
    if err == CL_INVALID_ARG_SIZE
        error("""
            Julia and OpenCL type don't match at kernel argument $idx: Found $T.
            Please make sure to define OpenCL structs correctly!
            You should be generally fine by using `__attribute__((packed))`, but sometimes the alignment of fields is different from Julia.
            Consider the following example:
                ```
                //packed
                // Tuple{NTuple{3, Float32}, Nothing, Float32}
                struct __attribute__((packed)) Test{
                    float3 f1;
                    int f2; // empty type gets replaced with Int32 (no empty types allowed in OpenCL)
                    // you might need to define the alignement of fields to match julia's layout
                    float f3; // for the types used here the alignement matches though!
                };
                // this is a case where Julia and OpenCL packed alignment would differ, so we need to specify it explicitely
                // Tuple{Int64, Int32}
                struct __attribute__((packed)) Test2{
                    long f1;
                    int __attribute__((aligned (8))) f2; // opencl would align this to 4 in packed layout, while Julia uses 8!
                };
                ```
            You can use `c.datatype_align(T)` to figure out the alignment of a Julia type!
        """)
    end
    @check err
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
        result2 = Vector{Int}(undef, 3)
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
                      wait_on::Union{Nothing,Vector{Event}}=nothing)
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
                        wait_on::Union{Nothing,Vector{Event}}=nothing)
    device = q[:device]
    max_work_dim = device[:max_work_item_dims]
    work_dim     = length(global_work_size)
    if work_dim > max_work_dim
        throw(ArgumentError("global_work_size has max dim of $max_work_dim"))
    end
    gsize = Vector{Csize_t}(undef, work_dim)
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
        goffset = Vector{Csize_t}(undef, work_dim)
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
        lsize = Vector{Csize_t}(undef, work_dim)
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

function info(k::Kernel, kinfo::Symbol)
    name(k::Kernel) = begin
        size = Ref{Csize_t}()
        @check api.clGetKernelInfo(k.id, CL_KERNEL_FUNCTION_NAME,
                                   0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
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
        result = Vector{CL_char}(undef, size[])
        @check api.clGetKernelInfo(k.id, CL_KERNEL_ATTRIBUTES,
                                   size[], result, size)
        return CLString(result)
    end

    info_map = Dict{Symbol, Function}(
        :name => name,
        :num_args => num_args,
        :reference_count => reference_count,
        :program => program,
        :attributes => attributes
    )

    try
        func = info_map[kinfo]
        func(k)
    catch err
        isa(err, KeyError) && error("OpenCL.Kernel has no info for: $kinfo")
        throw(err)
    end
end

#TODO set_arg sampler...
# OpenCL 1.2 function
#TODO: get_arg_info(k::Kernel, idx, param)
#TODO: enqueue_async_kernel()
