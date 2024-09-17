import LinearAlgebra

export CLArray, CLMatrix, CLVector, buffer

mutable struct CLArray{T, N} <: AbstractGPUArray{T, N}
    ctx::cl.Context

    data::DataRef{cl.SVMBuffer{UInt8}}
    offset::Int # offset in number of elements

    dims::NTuple{N, Int}

    # allocating constructor
    function CLArray{T,N}(::UndefInitializer, dims::Dims{N}; access=:rw) where {T,N}
        buf = cl.SVMBuffer{UInt8}(prod(dims) * sizeof(T), access)
        ref = DataRef(identity, buf)
        new(cl.context(), ref, 0, dims)
    end

    # low-level constructor for wrapping existing data
    function CLArray{T,N}(ref::DataRef{cl.SVMBuffer{UInt8}}, dims::Dims;
                          offset::Int=0) where {T,N}
        new(cl.context(), ref, offset, dims)
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
buffer(A::CLArray) = A.data[]

Base.elsize(::Type{<:CLArray{T}}) where {T} = sizeof(T)

Base.size(x::CLArray) = x.dims
Base.sizeof(x::CLArray) = Base.elsize(x) * length(x)

Base.unsafe_convert(::Type{Ptr{T}}, x::CLArray{T}) where {T} =
    convert(Ptr{T}, pointer(x.data[])) + x.offset*Base.elsize(x)

Base.:(==)(A::CLArray, B::CLArray) = Array(A) == Array(B)


## derived types

export DenseCLArray, DenseJLVector, DenseJLMatrix, DenseJLVecOrMat,
       StridedCLArray, StridedJLVector, StridedJLMatrix, StridedJLVecOrMat,
       AnyCLArray, AnyJLVector, AnyJLMatrix, AnyJLVecOrMat

# dense arrays: stored contiguously in memory
DenseCLArray{T,N} = CLArray{T,N}
DenseJLVector{T} = DenseCLArray{T,1}
DenseJLMatrix{T} = DenseCLArray{T,2}
DenseJLVecOrMat{T} = Union{DenseJLVector{T}, DenseJLMatrix{T}}

# strided arrays
StridedSubCLArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                            Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:CLArray,I}
StridedCLArray{T,N} = Union{CLArray{T,N}, StridedSubCLArray{T,N}}
StridedJLVector{T} = StridedCLArray{T,1}
StridedJLMatrix{T} = StridedCLArray{T,2}
StridedJLVecOrMat{T} = Union{StridedJLVector{T}, StridedJLMatrix{T}}

Base.pointer(x::StridedCLArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, x)
@inline function Base.pointer(x::StridedCLArray{T}, i::Integer) where T
    Base.unsafe_convert(Ptr{T}, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a CLArray
AnyCLArray{T,N} = Union{CLArray{T,N}, WrappedArray{T,N,CLArray,CLArray{T,N}}}
AnyJLVector{T} = AnyCLArray{T,1}
AnyJLMatrix{T} = AnyCLArray{T,2}
AnyJLVecOrMat{T} = Union{AnyJLVector{T}, AnyJLMatrix{T}}


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
    cl.enqueue_svm_fill(pointer(A), x, length(A))
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
  unsafe_copyto!(dest, doffs, src, soffs, n)
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
  unsafe_copyto!(dest, doffs, src, soffs, n; blocking=true)
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
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end
Base.copyto!(dest::CLArray{T}, src::CLArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

for (srcty, dstty) in [(:Array, :CLArray), (:CLArray, :Array), (:CLArray, :CLArray)]
    @eval begin
        function Base.unsafe_copyto!(dst::$dstty{T}, dst_off::Int,
                                     src::$srcty{T}, src_off::Int,
                                     N::Int; blocking::Bool=false) where T
            nbytes = N * sizeof(T)
            cl.enqueue_svm_memcpy(pointer(dst, dst_off), pointer(src, src_off), nbytes;
                                  blocking)
        end
        Base.unsafe_copyto!(dst::$dstty, src::$srcty, N; kwargs...) =
            unsafe_copyto!(dst, 1, src, 1, N; kwargs...)
    end
end


## broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct CLArrayStyle{N} <: AbstractGPUArrayStyle{N} end
CLArrayStyle{M}(::Val{N}) where {N,M} = CLArrayStyle{N}()

# identify the broadcast style of a (wrapped) array
BroadcastStyle(::Type{<:CLArray{T,N}}) where {T,N} = CLArrayStyle{N}()
BroadcastStyle(::Type{<:AnyCLArray{T,N}}) where {T,N} = CLArrayStyle{N}()

# allocation of output arrays
Base.similar(bc::Broadcasted{CLArrayStyle{N}}, ::Type{T}, dims) where {T,N} =
    similar(CLArray{T}, dims)


## regular gpu array adaptor

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CLArray}, xs::AT) where {AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CLArray, xs)

# if specific type parameters are specified, preserve those
Adapt.adapt_storage(::Type{<:CLArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CLArray{T}, xs)
Adapt.adapt_storage(::Type{<:CLArray{T, N}}, xs::AT) where {T, N, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CLArray{T,N}, xs)
