# GPUArrays.jl interface

function GPUArrays.derive(::Type{T}, a::CLArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.data)
    offset = a.offset + offset * sizeof(T)
    CLArray{T,N}(ref, dims; offset)
end
