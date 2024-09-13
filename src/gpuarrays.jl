# GPUArrays.jl interface


#
# Device functionality
#


## execution

struct CLArrayBackend <: AbstractGPUBackend end

struct CLKernelContext <: AbstractKernelContext end

@inline function GPUArrays.launch_heuristic(::CLArrayBackend, f::F, args::Vararg{Any,N};
                                             elements::Int, elements_per_thread::Int) where {F,N}
    kernel = @opencl launch=false f(CLKernelContext(), args...)
    wg_info = cl.work_group_info(kernel.fun, cl.device())

    # XXX: how many groups is a good number? the API doesn't tell us.
    #      measured on a low-end IGP, 32 blocks seems like a good sweet spot.
    #      note that this only matters for grid-stride kernels, like broadcast.
    return (threads=wg_info.size, blocks=32)
end

function GPUArrays.gpu_call(::CLArrayBackend, f, args, threads::Int, blocks::Int;
                            name::Union{String,Nothing})
    @opencl global_size=blocks*threads local_size=threads name=name f(CLKernelContext(), args...)
end


## on-device

# indexing

GPUArrays.blockidx(ctx::CLKernelContext) = get_group_id(1)
GPUArrays.blockdim(ctx::CLKernelContext) = get_local_size(1)
GPUArrays.threadidx(ctx::CLKernelContext) = get_local_id(1)
GPUArrays.griddim(ctx::CLKernelContext) = get_num_groups(1)

# math

@inline GPUArrays.cos(ctx::CLKernelContext, x) = cos(x)
@inline GPUArrays.sin(ctx::CLKernelContext, x) = sin(x)
@inline GPUArrays.sqrt(ctx::CLKernelContext, x) = sqrt(x)
@inline GPUArrays.log(ctx::CLKernelContext, x) = log(x)

# memory

@inline function GPUArrays.LocalMemory(::CLKernelContext, ::Type{T}, ::Val{dims}, ::Val{id}
                                      ) where {T, dims, id}
    ptr = SPIRVIntrinsics.emit_localmemory(Val(id), T, Val(prod(dims)))
    oneDeviceArray(dims, LLVMPtr{T, onePI.AS.Local}(ptr))
end

# synchronization

@inline GPUArrays.synchronize_threads(::CLKernelContext) = barrier()



#
# Host abstractions
#

GPUArrays.backend(::Type{<:CLArray}) = CLArrayBackend()

function GPUArrays.derive(::Type{T}, a::CLArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.data)
    offset = (a.offset * Base.elsize(a)) ÷ sizeof(T) + offset
    CLArray{T,N}(ref, dims; offset)
end

const GLOBAL_RNGs = Dict{cl.Device,GPUArrays.RNG}()
function GPUArrays.default_rng(::Type{<:CLArray})
    dev = cl.device()
    get!(GLOBAL_RNGs, dev) do
        N = dev.max_work_group_size
        state = CLArray{NTuple{4, UInt32}}(undef, N)
        rng = GPUArrays.RNG(state)
        Random.seed!(rng)
        rng
    end
end
