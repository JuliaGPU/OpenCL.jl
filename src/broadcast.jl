# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct CLArrayStyle{N, B} <: AbstractGPUArrayStyle{N} end
CLArrayStyle{M, B}(::Val{N}) where {N, M, B} = CLArrayStyle{N, B}()

# identify the broadcast style of a (wrapped) CLArray
BroadcastStyle(::Type{<:CLArray{T, N, B}}) where {T, N, B} = CLArrayStyle{N, B}()
BroadcastStyle(W::Type{<:WrappedCLArray{T, N}}) where {T, N} =
    CLArrayStyle{N, memtype(Adapt.unwrap_type(W))}()

# when we are dealing with different buffer styles, we cannot know
# which one is better, so use shared memory
BroadcastStyle(
    ::CLArrayStyle{N, B1},
    ::CLArrayStyle{N, B2}
) where {N, B1, B2} =
    CLArrayStyle{N, cl.UnifiedSharedMemory}()

# allocation of output arrays
Base.similar(bc::Broadcasted{CLArrayStyle{N, B}}, ::Type{T}, dims) where {T, N, B} =
    similar(CLArray{T, length(dims), B}, dims)
