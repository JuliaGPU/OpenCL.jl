import LinearAlgebra

export CLArray, CLMatrix, CLVector, to_host

mutable struct CLArray{T, N} <: CLObject
    ctx::cl.Context
    buffer::cl.Buffer{T}
    size::NTuple{N, Int}
end

const CLMatrix{T} = CLArray{T,2}
const CLVector{T} = CLArray{T,1}

## constructors

function CLArray(buf::cl.Buffer{T}, sz::Tuple{Vararg{Int}}) where T
    CLArray(cl.context(), buf, sz)
end

function CLArray(flags::Tuple{Vararg{Symbol}},
                 hostarray::AbstractArray{T,N}) where {T, N}
    buf = cl.Buffer(T, length(hostarray), flags, hostbuf=hostarray)
    sz = size(hostarray)
    CLArray(cl.context(), buf, sz)
end

CLArray(hostarray::AbstractArray{T,N};
        flags=(:rw, :copy)) where {T, N} = CLArray((:rw, :copy), hostarray)

Base.copy(A::CLArray; ctx=A.ctx,
          buffer=A.buffer, size=A.size) = CLArray(ctx, buffer, size)

function Base.deepcopy(A::CLArray{T,N}) where {T, N}
    new_buf = cl.Buffer(T, A.ctx, prod(A.size))
    copy!(new_buf, A.buffer)
    return CLArray(A.ctx, new_buf, A.size)
end

"""
Create in device memory array of type `t` and size `dims` filled by value `x`.
"""
function fill(::Type{T}, x::T, dims...) where T
    v = opencl_version(cl.context())
    if v.major == 1 && v.minor >= 2
        buf = cl.Buffer(T, prod(dims))
        fill!(q, buf, x)
    else
        buf = cl.Buffer(T, prod(dims), (:rw, :copy), hostbuf=Base.fill(x, dims))
    end
    return CLArray(buf, dims)
end

zeros(::Type{T}, dims...) where {T} = fill(T, T(0), dims...)
zeros(dims...) = fill(Float64, Float64(0), dims...)
ones(::Type{T}, dims...) where {T} = fill(T, T(1), dims...)
ones(dims...) = fill(Float64, Float64(1), dims...)


## core functions

buffer(A::CLArray) = A.buffer
Base.pointer(A::CLArray) = A.buffer.id
context(A::CLArray) = cl.context(A.buffer)
Base.eltype(A::CLArray{T, N}) where {T, N} = T
Base.size(A::CLArray) = A.size
Base.size(A::CLArray, dim::Integer) = A.size[dim]
Base.ndims(A::CLArray) = length(size(A))
Base.length(A::CLArray) = prod(size(A))
Base.:(==)(A:: CLArray, B:: CLArray) =
    buffer(A) == buffer(B) && size(A) == size(B)
function Base.reshape(A::CLArray, dims...)
    @assert prod(dims) == prod(size(A))
    return copy(A, size=dims)
end

## show

Base.show(io::IO, A::CLArray{T,N}) where {T, N} =
    print(io, "CLArray{$T,$N}($(buffer(A)),$(size(A)))")

## to_host

function to_host(A::CLArray{T,N}) where {T, N}
    hA = Array{T}(undef, size(A)...)
    copy!(hA, buffer(A))
    return hA
end

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
    ctx = context(A)
    kernel = get_kernel(ctx, TRANSPOSE_FLOAT_PROGRAM_PATH, "transpose",
                        block_size=block_size)
    h, w = size(A)
    lmem = cl.LocalMem(Float32, block_size * (block_size + 1))
    cl.set_args!(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem)
    return cl.enqueue_kernel(kernel, (h, w), (block_size, block_size))
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
    ctx = context(A)
    kernel = get_kernel(ctx, TRANSPOSE_DOUBLE_PROGRAM_PATH, "transpose",
                          block_size=block_size)
    h, w = size(A)
    # lmem = cl.LocalMem(Float64, block_size * (block_size + 1))
    lmem = cl.LocalMem(Float64, block_size * block_size)
    cl.set_args!(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem)
    return cl.enqueue_kernel(kernel, (h, w), (block_size, block_size))
end

"""Transpose CLMatrix A"""
function LinearAlgebra.transpose(A::CLMatrix{Float64})
    B = zeros(Float64, reverse(size(A))...)
    ev = LinearAlgebra.transpose!(B, A)
    cl.wait(ev)
    return B
end
