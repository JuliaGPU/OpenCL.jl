using Random

const GLOBAL_RNGs = Dict{cl.Device,GPUArrays.RNG{CLArray}}()
const global_rngs_lock = ReentrantLock()

# one RNG per device, used by the RNG-less `rand!`/`randn!` methods and `seed!`
function gpuarrays_rng()
    dev = cl.device()
    return Base.@lock global_rngs_lock begin
        get!(() -> GPUArrays.RNG{CLArray}(), GLOBAL_RNGs, dev)
    end
end

# GPUArrays in-place
Random.rand!(A::WrappedCLArray) = Random.rand!(gpuarrays_rng(), A)
Random.randn!(A::WrappedCLArray) = Random.randn!(gpuarrays_rng(), A)

# GPUArrays out-of-place
rand(T::Type, dims::Dims) = Random.rand!(CLArray{T}(undef, dims...))
randn(T::Type, dims::Dims; kwargs...) = Random.randn!(CLArray{T}(undef, dims...); kwargs...)

# support all dimension specifications
rand(T::Type, dim1::Integer, dims::Integer...) = Random.rand!(CLArray{T}(undef, dim1, dims...))
randn(T::Type, dim1::Integer, dims::Integer...; kwargs...) = Random.randn!(CLArray{T}(undef, dim1, dims...); kwargs...)

# untyped out-of-place
rand(dim1::Integer, dims::Integer...) = Random.rand!(CLArray{Float32}(undef, dim1, dims...))
randn(dim1::Integer, dims::Integer...; kwargs...) = Random.randn!(CLArray{Float32}(undef, dim1, dims...); kwargs...)

# seeding
seed!(seed = Base.rand(UInt64)) = Random.seed!(gpuarrays_rng(), seed)
