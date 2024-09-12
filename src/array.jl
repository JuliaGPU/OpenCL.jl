import LinearAlgebra

export CLArray, CLMatrix, CLVector, buffer

mutable struct CLArray{T, N} <: AbstractArray{T, N}
    ctx::cl.Context
    buffer::cl.Buffer{T}
    size::NTuple{N, Int}

    function CLArray{T,N}(::UndefInitializer, dims::Dims{N};
                          host=:rw, device=:rw) where {T,N}
        buf = cl.Buffer{T}(prod(dims); host, device)
        new(cl.context(), buf, dims)
    end

    function CLArray{T,N}(buf::cl.Buffer, dims::Dims) where {T,N}
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

buffer(A::CLArray) = A.buffer
Base.pointer(A::CLArray) = A.buffer.id
context(A::CLArray) = cl.context(A.buffer)
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
    buf = cl.Buffer(hostarray; kwargs...)
    sz = size(hostarray)
    CLArray{T,N}(buf, sz)
end

function Base.Array(A::CLArray{T,N}) where {T, N}
    hA = Array{T}(undef, size(A)...)
    copyto!(hA, A)
    return hA
end

function Base.cconvert(::Type{Ptr{T}}, A::CLArray{T}) where T
    buffer(A)
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


## other array operations

const TRANSPOSE_FLOAT_PROGRAM_PATH = joinpath(@__DIR__, "kernels", "transpose_float.cl")
const TRANSPOSE_DOUBLE_PROGRAM_PATH = joinpath(@__DIR__, "kernels", "transpose_double.cl")

function max_block_size(h::Int, w::Int)
    dim1, dim2 = cl.device().max_work_item_size[1:2]
    wgsize = cl.device().max_work_group_size
    wglimit = floor(Int, sqrt(wgsize))
    return gcd(dim1, dim2, h, w, wglimit)
end

"""
Transpose CLMatrix A, write result to a preallicated CLMatrix B
"""
function LinearAlgebra.transpose!(B::CLMatrix{Float32}, A::CLMatrix{Float32})
    block_size = max_block_size(size(A, 1), size(A, 2))
    kernel = get_kernel(TRANSPOSE_FLOAT_PROGRAM_PATH, "transpose",
                        block_size=block_size)
    h, w = size(A)
    lmem = cl.LocalMem(Float32, block_size * (block_size + 1))
    return cl.call(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem;
                   global_size=(h, w), local_size=(block_size, block_size))
end

"""Transpose CLMatrix A"""
function LinearAlgebra.transpose(A::CLMatrix{Float32})
    B = zeros(Float32, reverse(size(A))...)
    ev = LinearAlgebra.transpose!(B, A)
    wait(ev)
    return B
end

"""Transpose CLMatrix A, write result to a preallicated CLMatrix B"""
function LinearAlgebra.transpose!(B::CLMatrix{Float64}, A::CLMatrix{Float64})
    if !in("cl_khr_fp64", cl.device().extensions)
        throw(ArgumentError("Double precision not supported by device"))
    end
    block_size = max_block_size(size(A, 1), size(A, 2))
    kernel = get_kernel(TRANSPOSE_DOUBLE_PROGRAM_PATH, "transpose",
                        block_size=block_size)
    h, w = size(A)
    lmem = cl.LocalMem(Float32, block_size * (block_size + 1))
    return cl.call(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem;
                   global_size=(h, w), local_size=(block_size, block_size))
end

"""Transpose CLMatrix A"""
function LinearAlgebra.transpose(A::CLMatrix{Float64})
    B = zeros(Float64, reverse(size(A))...)
    ev = LinearAlgebra.transpose!(B, A)
    wait(ev)
    return B
end
