
type CLArray{T,N} <: AbstractArray{T,N}
    ctx::Context
    queue::CmdQueue
    buffer::Buffer{T}
    size::NTuple{N,Int}
end

@compat CLMatrix{T} = CLArray{T,2}
@compat CLVector{T} = CLArray{T,1}

## constructors

function CLArray{T}(buf::Buffer{T}, queue::CmdQueue, sz::Tuple{Vararg{Int}})
    ctx = context(buf)
    CLArray(ctx, queue, buf, sz)
end

function CLArray{T,N}(queue::CmdQueue,
                      flags::Tuple{Vararg{Symbol}},
                      hostarray::AbstractArray{T,N})
    ctx = context(queue)
    buf = Buffer(T, ctx, flags, hostbuf=hostarray)
    sz = size(hostarray)
    CLArray(ctx, queue, buf, sz)
end

function CLArray{T,N}(queue::CmdQueue, hostarray::AbstractArray{T,N};
                      flags=(:rw, :copy))
    CLArray(queue, (:rw, :copy), hostarray)
end

Base.copy(A::CLArray; ctx=A.ctx, queue=A.queue,
          buffer=A.buffer, size=A.size) =
    CLArray(ctx, queue, buffer, size)
Base.copy!{T}(dest::Array{T}, src::CLArray{T}; queue=src.queue)  = copy!(queue, dest, src.buffer)
Base.copy!{T}(dest::CLArray{T}, src::Array{T}; queue=dest.queue) = copy!(queue, dest.buffer, src)
function Base.deepcopy{T,N}(A::CLArray{T,N})
    new_buf = Buffer(T, A.ctx, prod(A.size))
    copy!(A.queue, new_buf, A.buffer)
    return CLArray(A.ctx, A.queue, new_buf, A.size)
end


"""
Create in device memory array of type `t` and size `dims` filled by value `x`.
"""
function Base.fill{T}(::Type{T}, q::CmdQueue, x::T, dims...)
    ctx = info(q, :context)
    v = opencl_version(ctx)
    if v.major == 1 && v.minor >= 2
        buf = Buffer(T, ctx, prod(dims))
        fill!(q, buf, x)
    else
        buf = Buffer(T, ctx, (:rw, :copy), prod(dims), hostbuf=fill(x, dims))
    end
    return CLArray(buf, q, dims)
end

Base.zeros{T}(::Type{T}, q::CmdQueue, dims...) = fill(T, q, T(0), dims...)
Base.zeros(q::CmdQueue, dims...) = fill(Float64, q, Float64(0), dims...)
Base.ones{T}(::Type{T}, q::CmdQueue, dims...) = fill(T, q, T(1), dims...)
Base.ones(q::CmdQueue, dims...) = fill(Float64, q, Float64(1), dims...)


## core functions

buffer(A::CLArray) = A.buffer
Base.pointer(A::CLArray) = A.buffer.id
context(A::CLArray) = context(A.buffer)
queue(A::CLArray) = A.queue
Base.size(A::CLArray) = A.size
Base.size(A::CLArray, dim::Integer) = A.size[dim]
Base.ndims(A::CLArray) = length(size(A))
Base.length(A::CLArray) = prod(size(A))
Base.:(==)(A:: CLArray, B:: CLArray) =
    buffer(A) == buffer(B) && size(A) == size(B)
Base.reshape(A::CLArray, dims::Tuple{Vararg{Int}}) = begin
    @assert prod(dims) == prod(size(A))
    return copy(A, size=dims)
end
Base.reshape(A::CLArray, dims::Int...) = reshape(A, dims)

## show

Base.show{T,N}(io::IO, A::CLArray{T,N}) =
    print(io, "CLArray{$T,$N}($(buffer(A)),$(size(A)))")

## to_host

function to_host{T,N}(A::CLArray{T,N}; queue=A.queue)
    hA = Array{T}(size(A))
    copy!(queue, hA, buffer(A))
    return hA
end

## other array operations

const TRANSPOSE_PROGRAM_PATH = joinpath(dirname(@__FILE__), "kernels/transpose.cl")

function max_block_size(queue::CmdQueue, h::Int, w::Int)
    dev = info(queue, :device)
    dim1, dim2 = info(dev, :max_work_item_size)[1:2]
    wgsize = info(dev, :max_work_group_size)
    wglimit = floor(Int, sqrt(wgsize))
    return gcd(dim1, dim2, h, w, wglimit)
end

"""Transpose CLMatrix A, write result to a preallicated CLMatrix B"""
function Base.transpose!(B::CLMatrix{Float32}, A::CLMatrix{Float32};
                         queue=A.queue)
    block_size = max_block_size(queue, size(A, 1), size(A, 2))
    ctx = context(A)
    kernel = get_kernel(ctx, TRANSPOSE_PROGRAM_PATH, "transpose",
                          block_size=block_size)
    h, w = size(A)
    lmem = LocalMem(Float32, block_size * (block_size + 1))
    set_args!(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem)
    return enqueue_kernel(queue, kernel, (h, w), (block_size, block_size))
end

"""Transpose CLMatrix A"""
function Base.transpose(A::CLMatrix{Float32};
                        queue=A.queue)
    B = zeros(Float32, queue, reverse(size(A))...)
    ev = transpose!(B, A, queue=queue)
    wait(ev)
    return B
end

"""Transpose CLMatrix A, write result to a preallicated CLMatrix B"""
function Base.transpose!(B::CLMatrix{Float64}, A::CLMatrix{Float64};
                         queue=A.queue)
    block_size = max_block_size(queue, size(A, 1), size(A, 2))
    ctx = context(A)
    kernel = get_kernel(ctx, TRANSPOSE_PROGRAM_PATH, "transpose_double",
                          block_size=block_size)
    h, w = size(A)
    # lmem = LocalMem(Float64, block_size * (block_size + 1))
    lmem = LocalMem(Float64, block_size * block_size)
    set_args!(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem)
    return enqueue_kernel(queue, kernel, (h, w), (block_size, block_size))
end

"""Transpose CLMatrix A"""
function Base.transpose(A::CLMatrix{Float64};
                        queue=A.queue)
    B = zeros(Float64, queue, reverse(size(A))...)
    ev = transpose!(B, A, queue=queue)
    wait(ev)
    return B
end

"""Conjugate transpose for reals just wraps transpose"""
Base.ctranspose{T<:Union{Float32,Float64}}(A::CLMatrix{T};
                        queue=A.queue, block_size=32) = transpose(A;
                        queue=queue, block_size=block_size);

"""Conjugate transpose! for reals just wraps transpose!"""
Base.ctranspose!{T<:Union{Float32,Float64}}(A::CLMatrix{T}, B::CLMatrix{T};
                        queue=A.queue, block_size=32) = transpose!(A, B;
                        queue=queue, block_size=block_size);

