# GPUArrays.jl interface

function GPUArrays.derive(::Type{T}, a::CLArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.data)
    # the selector bytes are indexed by element, so advance them here: the byte offset
    # can't tell how many elements of zero size (like unions of singletons) were skipped.
    # union arrays only derive arrays of the same element type, as `reinterpret` rejects them
    maxsize = Base.isbitsunion(T) ? a.maxsize + offset : a.maxsize
    offset = a.offset + offset * aligned_sizeof(T)
    CLArray{T,N}(ref, dims; maxsize, offset)
end
