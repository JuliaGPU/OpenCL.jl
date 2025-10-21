# OpenCL.Kernel

export clcall


## kernel object

mutable struct Kernel <: CLObject
    const id::cl_kernel

    function Kernel(k::cl_kernel, retain::Bool=false)
        kernel = new(k)
        retain && clRetainKernel(kernel)
        finalizer(clReleaseKernel, kernel)
        return kernel
    end
end

Base.unsafe_convert(::Type{cl_kernel}, k::Kernel) = k.id

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
        Int(get(CL_KERNEL_LOCAL_MEM_SIZE, cl_ulong))
    elseif s == :private_mem_size
        Int(get(CL_KERNEL_PRIVATE_MEM_SIZE, cl_ulong))
    elseif s == :prefered_size_multiple
        Int(get(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, Csize_t))
    else
        getfield(ki, s)
    end
end

struct KernelSubGroupInfo
    kernel::Kernel
    device::Device
    local_work_size::Vector{Csize_t}
end
sub_group_info(k::Kernel, d::Device, l) = KernelSubGroupInfo(k, d, Vector{Csize_t}(l))

# Helper function for getting local size for a specific sub-group count
function local_size_for_sub_group_count(ki::KernelSubGroupInfo, sub_group_count::Integer)
    k = getfield(ki, :kernel)
    d = getfield(ki, :device)
    input_value = Ref{Csize_t}(sub_group_count)
    result = Ref{NTuple{3, Csize_t}}()
    clGetKernelSubGroupInfo(k, d, CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT,
                           sizeof(Csize_t), input_value, sizeof(NTuple{3, Csize_t}), result, C_NULL)
    return Int.(result[])
end

function Base.getproperty(ki::KernelSubGroupInfo, s::Symbol)
    k = getfield(ki, :kernel)
    d = getfield(ki, :device)
    lws = getfield(ki, :local_work_size)

    function get(val, typ)
        result = Ref{typ}()
        clGetKernelSubGroupInfo(k, d, val, sizeof(lws), lws, sizeof(typ), result, C_NULL)
        return result[]
    end

    if s == :max_sub_group_size
        Int(get(CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, Csize_t))
    elseif s == :sub_group_count
        Int(get(CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE, Csize_t))
    elseif s == :local_size_for_sub_group_count
        # This requires input_value to be the desired sub-group count
        error("local_size_for_sub_group_count requires specifying desired sub-group count")
    elseif s == :max_num_sub_groups
        Int(get(CL_KERNEL_MAX_NUM_SUB_GROUPS, Csize_t))
    elseif s == :compile_num_sub_groups
        Int(get(CL_KERNEL_COMPILE_NUM_SUB_GROUPS, Csize_t))
    elseif s == :compile_sub_group_size
        Int(get(CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL, Csize_t))
    else
        getfield(ki, s)
    end
end


## kernel calling

function enqueue_kernel(k::Kernel, global_work_size, local_work_size=nothing;
                        global_work_offset=nothing, wait_on::Vector{Event}=Event[],
                        rng_state=false, nargs=nothing)
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

    if rng_state
        if local_work_size !== nothing
            num_sub_groups = KernelSubGroupInfo(k, device(), lsize).sub_group_count
        else
            num_sub_groups = KernelSubGroupInfo(k, device(), Csize_t[]).max_num_sub_groups
        end
        if nargs === nothing
            nargs = k.num_args - 2
        end
        rng_state_size = sizeof(UInt32) * num_sub_groups
        set_arg!(k, nargs + 1, LocalMem(UInt32, rng_state_size))
        set_arg!(k, nargs + 2, LocalMem(UInt32, rng_state_size))
    end

    if !isempty(wait_on)
        n_events = length(wait_on)
        wait_event_ids = [evt.id for evt in wait_on]
    else
        n_events = 0
        wait_event_ids = C_NULL
    end

    ret_event = Ref{cl_event}()
    clEnqueueNDRangeKernel(queue(), k, work_dim, goffset, gsize, lsize,
                           n_events, wait_event_ids, ret_event)
    return Event(ret_event[], retain=false)
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

function call(
        k::Kernel, args...; global_size = (1,), local_size = nothing,
        global_work_offset = nothing, wait_on::Vector{Event} = Event[],
        indirect_memory::Vector{AbstractMemory} = AbstractMemory[],
        rng_state=false,
    )
    set_args!(k, args...)
    if !isempty(indirect_memory)
        svm_pointers = CLPtr{Cvoid}[]
        usm_pointers = CLPtr{Cvoid}[]
        bda_pointers = CLPtr{Cvoid}[]
        device_access = host_access = shared_access = false
        for memory in indirect_memory
            ptr = pointer(memory)
            if ptr == C_NULL || ptr == CL_NULL
                continue
            end

            if memory isa SharedVirtualMemory
                push!(svm_pointers, ptr)
            elseif memory isa Buffer
                push!(bda_pointers, ptr)
            elseif memory isa UnifiedDeviceMemory
                device_access = true
                push!(usm_pointers, ptr)
            elseif memory isa UnifiedHostMemory
                host_access = true
                push!(usm_pointers, reinterpret(CLPtr{Cvoid}, ptr))
            elseif memory isa UnifiedSharedMemory
                shared_access = true
                push!(usm_pointers, ptr)
            else
                throw(ArgumentError("Unknown memory type"))
            end
        end

        # configure USM access
        if device_access
            clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL, sizeof(cl_bool), Ref{cl_bool}(true))
        end
        if host_access
            clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL, sizeof(cl_bool), Ref{cl_bool}(true))
        end
        if shared_access
            clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL, sizeof(cl_bool), Ref{cl_bool}(true))
        end

        # set the pointers
        if !isempty(svm_pointers)
            clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_SVM_PTRS, sizeof(svm_pointers), svm_pointers)
        end
        if !isempty(bda_pointers)
            clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT, sizeof(bda_pointers), bda_pointers)
        end
        if !isempty(usm_pointers)
            clSetKernelExecInfo(k, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL, sizeof(usm_pointers), usm_pointers)
        end
    end
    enqueue_kernel(k, global_size, local_size; global_work_offset, wait_on, rng_state, nargs=length(args))
end

# From `julia/base/reflection.jl`, adjusted to add specialization on `t`.
function _to_tuple_type(t)
    if isa(t, Tuple) || isa(t, AbstractArray) || isa(t, SimpleVector)
        t = Tuple{t...}
    end
    if isa(t, Type) && t <: Tuple
        for p in (Base.unwrap_unionall(t)::DataType).parameters
            if isa(p, Core.TypeofVararg)
                p = Base.unwrapva(p)
            end
            if !(isa(p, Type) || isa(p, TypeVar))
                error("argument tuple type must contain only types")
            end
        end
    else
        error("expected tuple type")
    end
    t
end

clcall(f::F, types::Tuple, args::Vararg{Any,N}; kwargs...) where {N,F} =
    clcall(f, _to_tuple_type(types), args...; kwargs...)

function clcall(k::Kernel, types::Type{T}, args::Vararg{Any,N}; kwargs...) where {T,N}
    call_closure = function (converted_args::Vararg{Any,N})
        call(k, converted_args...; kwargs...)
    end
    convert_arguments(call_closure, types, args...)
end


## generic argument conversion

# convert the argument values to match the kernel's signature (specified by the user)
# (this mimics `lower-ccall` in julia-syntax.scm)
clconvert(typ, arg) = Base.cconvert(typ, arg)
unsafe_clconvert(typ, arg) = Base.unsafe_convert(typ, arg)
@inline @generated function convert_arguments(f::Function, ::Type{tt}, args...) where {tt}
    types = tt.parameters

    ex = quote end

    converted_args = Vector{Symbol}(undef, length(args))
    arg_ptrs = Vector{Symbol}(undef, length(args))
    for i in 1:length(args)
        converted_args[i] = gensym()
        arg_ptrs[i] = gensym()
        push!(ex.args, :($(converted_args[i]) = clconvert($(types[i]), args[$i])))
        push!(ex.args, :($(arg_ptrs[i]) = unsafe_clconvert($(types[i]), $(converted_args[i]))))
    end

    append!(ex.args, (quote
        GC.@preserve $(converted_args...) begin
            f($(arg_ptrs...))
        end
    end).args)

    return ex
end

function set_args!(k::Kernel, args...)
    for (i, a) in enumerate(args)
        set_arg!(k, i, a)
    end
end

function set_arg!(k::Kernel, idx::Integer, arg::T) where {T}
    ref = Ref(arg)
    tsize = sizeof(ref)
    err = unchecked_clSetKernelArg(k, idx - 1, tsize, ref)
    if err == CL_INVALID_ARG_SIZE
        error("""Mismatch between Julia and OpenCL type for kernel argument $idx.

                 Possible reasons:
                 - OpenCL does not support empty types.
                 - Vectors of length 3 (e.g., `float3`) are packed as 4-element vectors;
                   consider padding your tuples.
                 - The alignment of fields in your struct may not match the OpenCL layout.
                   Make sure your Julia definition matches the OpenCL layout, e.g., by
                   using `__attribute__((packed))` in your OpenCL struct definition.""")
    elseif err != CL_SUCCESS
        throw(CLError(err))
    end
    return k
end

function set_arg!(k::Kernel, idx::Integer, arg::Nothing)
    @assert idx > 0
    clSetKernelArg(k, idx - 1, sizeof(cl_mem), C_NULL)
end


## memory arguments

# passing pointers directly requires the memory type to be specified
set_arg!(k::Kernel, idx::Integer, arg::Union{Ptr, CLPtr}) =
    error("""Cannot pass a pointer to a kernel without specifying the origin memory type.
             Pass a memory object instead, or use the 4-arg version of `set_arg!` to indicate the memory type.""")
function set_arg!(k::Kernel, idx::Integer, ptr::Union{Ptr, CLPtr}, typ::Type)
    # XXX: this assumes that the receiving argument is pointer-typed, which is not the case
    #      with Julia's `Ptr` ABI. Instead, one should reinterpret the pointer as a
    #      `Core.LLVMPtr`, which _is_ pointer-valued. We retain this handling for `Ptr` for
    #      users passing pointers to OpenCL C, and because `Ptr` is pointer-valued starting
    #      with Julia 1.12.
    if typ == SharedVirtualMemory
        clSetKernelArgSVMPointer(k, idx - 1, ptr)
    elseif typ <: UnifiedMemory
        clSetKernelArgMemPointerINTEL(k, idx - 1, ptr)
    elseif typ == Buffer
        # XXX: this branch is never taken, as we currently still use plain `clSetKernelArg`,
        #      which is only possible because our pointer always comes from a `Buffer`.
        clSetKernelArgDevicePointerEXT(k, idx - 1, ptr)
    else
        error("Unknown memory type")
    end
end

# memory objects: pass the memory object itself
unsafe_clconvert(typ::Type{<:Union{Ptr, CLPtr}}, mem::AbstractMemoryObject) = mem
function set_arg!(k::Kernel, idx::Integer, arg::AbstractMemoryObject)
    arg_boxed = Ref(arg.id)
    clSetKernelArg(k, idx - 1, sizeof(cl_mem), arg_boxed)
end

# raw memory: pass as a pointer, keeping track of the memory type
struct TrackedPtr{T,M}
    ptr::Union{Ptr{T}, CLPtr{T}}
end
unsafe_clconvert(typ::Type{<:Union{Ptr{T}, CLPtr{T}}}, mem::AbstractPointerMemory) where {T} =
    TrackedPtr{T,typeof(mem)}(Base.unsafe_convert(typ, mem))
set_arg!(k::Kernel, idx::Integer, arg::TrackedPtr{<:Any,M}) where {M} =
    set_arg!(k, idx, arg.ptr, M)
set_arg!(k::Kernel, idx::Integer, arg::AbstractPointerMemory) =
    set_arg!(k, idx, pointer(arg), typeof(arg))


## local memory arguments

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
Base.sizeof(l::LocalMem{T}) where {T} = Int(l.nbytes)
Base.length(l::LocalMem{T}) where {T} = Int(l.nbytes ÷ sizeof(T))

# XXX: can we avoid the `set_arg!` special case and `clconvert` to `CU_NULL`?
#      the problem is the size being passed to `clSetKernelArg`
unsafe_clconvert(::Type{CLPtr{T}}, l::LocalMem{T}) where {T} = l
function set_arg!(k::Kernel, idx::Integer, arg::LocalMem)
    clSetKernelArg(k, idx - 1, arg.nbytes, C_NULL)
end
