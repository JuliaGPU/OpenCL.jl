# OpenCL.Kernel
mutable struct Kernel <: CLObject
    id::cl_kernel

    function Kernel(k::cl_kernel, retain=false)
        if retain
            clRetainKernel(k)
        end
        kernel = new(k)
        finalizer(_finalize, kernel)
        return kernel
    end
end

function _finalize(k::Kernel)
    if k.id != C_NULL
        clReleaseKernel(k.id)
        k.id = C_NULL
    end
end

Base.unsafe_convert(::Type{cl_kernel}, k::Kernel) = k.id

Base.pointer(k::Kernel) = k.id

Base.show(io::IO, k::Kernel) = begin
    print(io, "OpenCL.Kernel(\"$(k.function_name)\" nargs=$(k.num_args))")
end

function Kernel(p::Program, kernel_name::String)
    for (dev, status) in p.build_status
        if status != CL_BUILD_SUCCESS
            msg = "OpenCL.Program has to be built before Kernel constructor invoked"
            throw(ArgumentError(msg))
        end
    end
    err_code = Ref{Cint}()
    kernel_id = clCreateKernel(p, kernel_name, err_code)
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
    clSetKernelArg(k, cl_uint(idx-1), sizeof(cl_mem), C_NULL)
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
    clSetKernelArg(k, cl_uint(idx-1), sizeof(cl_mem), arg_boxed)
    return k
end

function set_arg!(k::Kernel, idx::Integer, arg::LocalMem)
    @assert idx > 0 "Kernel idx must be bigger 0"
    clSetKernelArg(k, cl_uint(idx-1), arg.nbytes, C_NULL)
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
    err = unchecked_clSetKernelArg(k, cl_uint(idx - 1), tsize, ref)
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
    if err != CL_SUCCESS
        throw(CLError(err))
    end
    return k
end

function set_args!(k::Kernel, args...)
    for (i, a) in enumerate(args)
        set_arg!(k, i, a)
    end
end

function enqueue_kernel(k::Kernel, global_work_size, local_work_size=nothing;
                        global_work_offset=nothing, wait_on::Vector{Event}=Event[])
    max_work_dim = device().max_work_item_dims
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
    else
        # null global offset means (0, 0, 0)
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
    else
        # null local size means OpenCL decides
    end

    if !isempty(wait_on)
        n_events = cl_uint(length(wait_on))
        wait_event_ids = [evt.id for evt in wait_on]
    else
        n_events = cl_uint(0)
        wait_event_ids = C_NULL
    end

    ret_event = Ref{cl_event}()
    clEnqueueNDRangeKernel(queue(), k, cl_uint(work_dim), goffset, gsize, lsize,
                           n_events, wait_event_ids, ret_event)
    return Event(ret_event[], retain=false)
end

function call(k::Kernel, args...; global_size=(1,), local_size=nothing,
              global_work_offset=nothing, wait_on::Vector{Event}=Event[])
    set_args!(k, args...)
    enqueue_kernel(k, global_size, local_size; global_work_offset, wait_on)
end

function enqueue_task(k::Kernel; wait_for=nothing)
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
    ret_event = Ref{cl_event}()
    clEnqueueTask(queue(), k, n_evts, evt_ids, ret_event)
    return ret_event[]
end

function Base.getproperty(k::Kernel, s::Symbol)
    if s == :function_name
        size = Ref{Csize_t}()
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, 0, C_NULL, size)
        result = Vector{Cchar}(undef, size[])
        clGetKernelInfo(k, CL_KERNEL_FUNCTION_NAME, size[], result, C_NULL)
        return GC.@preserve result unsafe_string(pointer(result))
    elseif s == :num_args
        result = Ref{Cuint}()
        clGetKernelInfo(k, CL_KERNEL_NUM_ARGS, sizeof(Cuint), result, C_NULL)
        return Int(result[])
    elseif s == :reference_count
        result = Ref{Cuint}()
        clGetKernelInfo(k, CL_KERNEL_REFERENCE_COUNT, sizeof(Cuint), result, C_NULL)
        return Int(result[])
    elseif s == :context
        result = Ref{cl_context}()
        clGetKernelInfo(k, CL_KERNEL_CONTEXT, sizeof(cl_context), result, C_NULL)
        return Context(result[], retain=true)
    elseif s == :program
        result = Ref{cl_program}()
        clGetKernelInfo(k, CL_KERNEL_PROGRAM, sizeof(cl_program), result, C_NULL)
        return Program(result[], retain=true)
    elseif s == :attributes
        size = Ref{Csize_t}()
        err = unchecked_clGetKernelInfo(k, CL_KERNEL_ATTRIBUTES, 0, C_NULL, size)
        if err == CL_SUCCESS && size[] > 1
            result = Vector{Cchar}(undef, size[])
            clGetKernelInfo(k, CL_KERNEL_ATTRIBUTES, size[], result, C_NULL)
            return GC.@preserve result unsafe_string(pointer(result))
        else
            return ""
        end
    else
        return getfield(k, s)
    end
end

struct KernelWorkGroupInfo
    kernel::Kernel
    device::Device
end
work_group_info(k::Kernel, d::Device) = KernelWorkGroupInfo(k, d)

function Base.getproperty(ki::KernelWorkGroupInfo, s::Symbol)
    k = getfield(ki, :kernel)
    d = getfield(ki, :device)

    function get(val, typ)
        result = Ref{typ}()
        clGetKernelWorkGroupInfo(k, d, val, sizeof(typ), result, C_NULL)
        return result[]
    end

    if s == :size
        Int(get(CL_KERNEL_WORK_GROUP_SIZE, Csize_t))
    elseif s == :compile_size
        Int.(get(CL_KERNEL_COMPILE_WORK_GROUP_SIZE, NTuple{3, Csize_t}))
    elseif s == :local_mem_size
        Int(get(CL_KERNEL_LOCAL_MEM_SIZE, Culong))
    elseif s == :private_mem_size
        Int(get(CL_KERNEL_PRIVATE_MEM_SIZE, Culong))
    elseif s == :prefered_size_multiple
        Int(get(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, Csize_t))
    else
        getfield(ki, s)
    end
end

#TODO set_arg sampler...
# OpenCL 1.2 function
#TODO: get_arg_info(k::Kernel, idx, param)
#TODO: enqueue_async_kernel()
