using Random

gpuarrays_rng() = GPUArrays.default_rng(CLArray)

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
