import LinearAlgebra

export CLArray, CLMatrix, CLVector, buffer

mutable struct CLArray{T, N} <: AbstractGPUArray{T, N}
    ctx::cl.Context
    buffer::cl.SVMBuffer{T} # XXX: support regular buffers too?
    size::NTuple{N, Int}

    function CLArray{T,N}(::UndefInitializer, dims::Dims{N}; access=:rw) where {T,N}
        buf = cl.SVMBuffer{T}(prod(dims), access)
        new(cl.context(), buf, dims)
    end

    function CLArray{T,N}(buf::cl.SVMBuffer, dims::Dims) where {T,N}
        new(cl.context(), buf, dims)
    end
end


## convenience constructors

const CLMatrix{T} = CLArray{T,2}
const CLVector{T} = CLArray{T,1}

# type and dimensionality specified
CLArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}; kwargs...) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims); kwargs...)
CLArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}; kwargs...) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims); kwargs...)

# type but not dimensionality specified
CLArray{T}(::UndefInitializer, dims::NTuple{N,Integer}; kwargs...) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims); kwargs...)
CLArray{T}(::UndefInitializer, dims::Vararg{Integer,N}; kwargs...) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims); kwargs...)

# empty vector constructor
CLArray{T,1}() where {T} = CLArray{T,1}(undef, 0)

Base.similar(a::CLArray{T,N}; kwargs...) where {T,N} =
  CLArray{T,N}(undef, size(a); kwargs...)
Base.similar(a::CLArray{T}, dims::Base.Dims{N}; kwargs...) where {T,N} =
  CLArray{T,N}(undef, dims; kwargs...)
Base.similar(a::CLArray, ::Type{T}, dims::Base.Dims{N}; kwargs...) where {T,N} =
  CLArray{T,N}(undef, dims; kwargs...)

function Base.copy(a::CLArray{T,N}; kwargs...) where {T,N}
  b = similar(a; kwargs...)
  @inbounds copyto!(b, a)
end

function Base.deepcopy_internal(x::CLArray, dict::IdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end


## array interface

context(A::CLArray) = A.ctx
buffer(A::CLArray) = A.buffer

Base.pointer(A::CLArray, i::Integer=1) = pointer(buffer(A), i)
Base.eltype(A::CLArray{T, N}) where {T, N} = T
Base.size(A::CLArray) = A.size
Base.size(A::CLArray, dim::Integer) = A.size[dim]
Base.ndims(A::CLArray) = length(size(A))
Base.length(A::CLArray) = prod(size(A))
Base.:(==)(A:: CLArray, B:: CLArray) = buffer(A) == buffer(B) && size(A) == size(B)

function Base.reshape(A::CLArray{T}, dims::NTuple{N,Int}) where {T,N}
    @assert prod(dims) == prod(size(A))
    CLArray{T,N}(buffer(A), dims)
end


## conversions

function CLArray(hostarray::AbstractArray{T,N}; kwargs...) where {T, N}
    arr = CLArray{T,N}(undef, size(hostarray); kwargs...)
    copyto!(arr, hostarray)
    return arr
end

function Base.Array(A::CLArray{T,N}) where {T, N}
    hA = Array{T}(undef, size(A)...)
    copyto!(hA, A)
    return hA
end

function Base.cconvert(::Type{Ptr{T}}, A::CLArray{T}) where T
    buffer(A)
end

function Adapt.adapt_storage(to::KernelAdaptor, xs::CLArray{T,N}) where {T,N}
    ptr = adapt(to, buffer(xs))
    CLDeviceArray{T,N,AS.Global}(size(xs), reinterpret(LLVMPtr{T,AS.Global}, ptr))
end


## utilities

"""
Create in device memory array of type `t` and size `dims` filled by value `x`.
"""
function fill(x::T, dims) where T
    A = CLArray{T}(undef, dims)
    fill!(A, x)
end
fill(x, dims...) = fill(x, (dims...,))

function Base.fill!(A::CLArray{T}, x::T) where {T}
    cl.unsafe_fill!(buffer(A), x, length(A))
    A
end

zeros(::Type{T}, dims...) where {T} = fill(zero(T), dims...)
zeros(dims...) = fill(Float64(0), dims...)
ones(::Type{T}, dims...) where {T} = fill(one(T), dims...)
ones(dims...) = fill(Float64(1), dims...)


## memory copying

typetagdata(a::Array, i=1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
typetagdata(a::CLArray, i=1) =
  convert(ZePtr{UInt8}, a.data[]) + a.maxsize + a.offset + i - 1

function Base.copyto!(dest::CLArray{T}, doffs::Int, src::Array{T}, soffs::Int,
                      n::Int) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(buffer(dest), doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::CLArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::Array{T}, doffs::Int, src::CLArray{T}, soffs::Int,
                      n::Int) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, buffer(src), soffs, n; blocking=true)
  return dest
end
Base.copyto!(dest::Array{T}, src::CLArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::CLArray{T}, doffs::Int, src::CLArray{T}, soffs::Int,
                      n::Int) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  @assert context(dest) == context(src)
  unsafe_copyto!(buffer(dest), doffs, buffer(src), soffs, n)
  return dest
end

Base.copyto!(dest::CLArray{T}, src::CLArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))


## show

Base.show(io::IO, A::CLArray{T,N}) where {T, N} =
    print(io, "CLArray{$T,$N}($(buffer(A)),$(size(A)))")
