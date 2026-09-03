module OpenCLKernels

using ..OpenCL
using ..OpenCL: @device_override, method_table, kernel_convert, clfunction

import KernelInterface as KI

import SPIRVIntrinsics

import StaticArrays

import Adapt


## Back-end Definition

export OpenCLBackend

struct OpenCLBackend <: KI.GPU
end

KI.versioninfo(io::IO, ::OpenCLBackend) = OpenCL.versioninfo(io)

function KI.allocate(::OpenCLBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T
    if unified
        memory_backend = cl.unified_memory_backend()
        if memory_backend === cl.USMBackend()
            return CLArray{T, length(dims), cl.UnifiedSharedMemory}(undef, dims)
        elseif memory_backend === cl.SVMBackend()
            return CLArray{T, length(dims), cl.SharedVirtualMemory}(undef, dims)
        else
            throw(ArgumentError("Unified memory not supported"))
        end
    else
        return CLArray{T}(undef, dims)
    end
end

KI.supports_unified(::OpenCLBackend) = cl.default_memory_backend(cl.device(); unified=true) !== nothing

KI.get_backend(::CLArray) = OpenCLBackend()
# TODO should be non-blocking
KI.synchronize(::OpenCLBackend) = cl.finish(cl.queue())
KI.supports_float64(::OpenCLBackend) = in("cl_khr_fp64", cl.device().extensions)

Adapt.adapt_storage(::OpenCLBackend, a::Array) = Adapt.adapt(CLArray, a)
Adapt.adapt_storage(::OpenCLBackend, a::CLArray) = a
# Adapt.adapt_storage(::KI.CPU, a::CLArray) = convert(Array, a)


## Memory Operations

function KI.copyto!(::OpenCLBackend, A, B)
    copyto!(A, B)
    # TODO: Address device to host copies in jl being synchronizing
end


## Kernel Launch


function threads_to_workgroupsize(threads, ndrange)
    total = Ref(1)
    return map(ndrange) do n
        x = min(div(threads, total[]), n)
        total[] *= x
        return x
    end
end

KI.argconvert(::OpenCLBackend, arg) = kernel_convert(arg)

function KI.kernel_function(::OpenCLBackend, f::F, tt::TT=Tuple{}; name = nothing, kwargs...) where {F,TT}
    kern = clfunction(f, tt; name, kwargs...)
    KI.Kernel{OpenCLBackend, typeof(kern)}(OpenCLBackend(), kern)
end

function (obj::KI.Kernel{OpenCLBackend})(args...; numworkgroups=(), workgroupsize=(), ndrange=(), max_work_group_size=typemax(Int))
    KI.check_launch_args(numworkgroups, workgroupsize, ndrange)
    prod(ndrange) == 0 && return nothing

    numworkgroups, workgroupsize = KI.auto_launch_sizes(obj, numworkgroups, workgroupsize, ndrange, max_work_group_size)
    local_size = (workgroupsize..., ntuple(_ -> 1, 3 - length(workgroupsize))...)
    numworkgroups = (numworkgroups..., ntuple(_ -> 1, 3 - length(numworkgroups))...)
    global_size = local_size .* numworkgroups

    obj.kern(args...; local_size, global_size)
    return nothing
end


function KI.kernel_max_work_group_size(kernel::KI.Kernel{<:OpenCLBackend}; max_work_items::Int=typemax(Int))::Int
    wginfo = cl.work_group_info(kernel.kern.fun, cl.device())
    Int(min(wginfo.size, max_work_items))
end
function KI.max_work_group_size(::OpenCLBackend)::Int
    Int(cl.device().max_work_group_size)
end
function KI.sub_group_size(::OpenCLBackend)::Int
    cl.sub_group_size(cl.device())
end
function KI.multiprocessor_count(::OpenCLBackend)::Int
    Int(cl.device().max_compute_units)
end

function KI.shfl_down_types(::OpenCLBackend)
    backend_extensions = cl.device().extensions
    "cl_khr_subgroup_shuffle" in backend_extensions || return DataType[]

    res = copy(SPIRVIntrinsics.gentypes)

    if "cl_khr_fp64" ∉ backend_extensions
        res = setdiff(res, [Float64])
    end
    if "cl_khr_fp16" ∉ backend_extensions
        res = setdiff(res, [Float16])
    end

    return res
end

## Indexing Functions
## COV_EXCL_START

@device_override @inline function KI.get_local_id()
    return (; x = Int(get_local_id(1)), y = Int(get_local_id(2)), z = Int(get_local_id(3)))
end

@device_override @inline function KI.get_group_id()
    return (; x = Int(get_group_id(1)), y = Int(get_group_id(2)), z = Int(get_group_id(3)))
end

@device_override @inline function KI.get_global_id()
    return (; x = Int(get_global_id(1)), y = Int(get_global_id(2)), z = Int(get_global_id(3)))
end

@device_override @inline function KI.get_local_size()
    return (; x = Int(get_local_size(1)), y = Int(get_local_size(2)), z = Int(get_local_size(3)))
end

@device_override @inline function KI.get_num_groups()
    return (; x = Int(get_num_groups(1)), y = Int(get_num_groups(2)), z = Int(get_num_groups(3)))
end

@device_override @inline function KI.get_global_size()
    return (; x = Int(get_global_size(1)), y = Int(get_global_size(2)), z = Int(get_global_size(3)))
end

@device_override KI.get_sub_group_size() = get_sub_group_size()

@device_override KI.get_max_sub_group_size() = get_max_sub_group_size()

@device_override KI.get_num_sub_groups() = get_num_sub_groups()

@device_override KI.get_sub_group_id() = get_sub_group_id()

@device_override KI.get_sub_group_local_id() = get_sub_group_local_id()

## Shared and Scratch Memory

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ptr = OpenCL.emit_localmemory(T, Val(prod(Dims)))
    CLDeviceArray(Dims, ptr)
end

## Synchronization and Printing

@device_override @inline function KI.barrier()
    work_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
end

@device_override @inline function KI.sub_group_barrier()
    sub_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
end

@device_override function KI.shfl_down(val::T, offset::Integer) where T
    sub_group_shuffle(val, get_sub_group_local_id() + offset)
end

@device_override @inline function KI._print(args...)
    OpenCL._print(args...)
end
## COV_EXCL_STOP

end
