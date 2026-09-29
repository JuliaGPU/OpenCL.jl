# broadcasting

import Base.Broadcast: BroadcastStyle, Broadcasted

struct CLArrayStyle{N, B} <: AbstractGPUArrayStyle{N} end
CLArrayStyle{M, B}(::Val{N}) where {N, M, B} = CLArrayStyle{N, B}()

# identify the broadcast style of a (wrapped) CLArray
BroadcastStyle(::Type{<:CLArray{T, N, B}}) where {T, N, B} = CLArrayStyle{N, B}()
BroadcastStyle(W::Type{<:WrappedCLArray{T, N}}) where {T, N} =
    CLArrayStyle{N, memtype(Adapt.unwrap_type(W))}()

# when we are dealing with different buffer styles, we cannot know which one is better,
# so use memory that is accessible from both the host and the device: shared memory if
# either side uses USM (so the device supports it), and SVM otherwise (only buffers and
# SVM are mixed, as happens with host memory wrapped using `unsafe_wrap`)
merge_memtype(B1, B2) =
    B1 == B2 ? B1 :
    (B1 <: cl.UnifiedMemory || B2 <: cl.UnifiedMemory) ? cl.UnifiedSharedMemory :
    cl.SharedVirtualMemory
BroadcastStyle(
    ::CLArrayStyle{M, B1},
    ::CLArrayStyle{N, B2}
) where {M, N, B1, B2} =
    CLArrayStyle{max(M, N), merge_memtype(B1, B2)}()

# allocation of output arrays
Base.similar(bc::Broadcasted{CLArrayStyle{N, B}}, ::Type{T}, dims) where {T, N, B} =
    similar(CLArray{T, length(dims), B}, dims)
