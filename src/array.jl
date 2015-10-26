
type CLArray{T,N} <: CLObject
    ctx::Context
    queue::CmdQueue
    buffer::Buffer{T}
    size::NTuple{N,Int}
end

typealias CLMatrix{T} CLArray{T,2}
typealias CLVector{T} CLArray{T,1}

##  constructors

@compat function CLArray{T,N}(ctx::Context,
                              queue::CmdQueue,
                              flags::Tuple{Vararg{Symbol}},
                              hostarray::AbstractArray{T,N})
    buf = Buffer(T, ctx, flags, hostbuf=hostarray)
    sz = size(hostarray)
    CLArray(ctx, queue, buf, sz)
end

function CLArray{T,N}(ctx::Context, hostarray::AbstractArray{T,N};
                      queue=CmdQueue(ctx), flags=(:rw, :copy))
    CLArray(ctx, CmdQueue(ctx), (:rw, :copy), hostarray)
end

@compat function CLArray{T}(buf::Buffer{T}, sz::Tuple{Vararg{Int}})
    ctx = context(buf)
    queue = CmdQueue(ctx)
    CLArray(context(buf), queue, buf, sz)
end

Base.copy(A::CLArray; ctx=A.ctx, queue=A.queue, buffer=A.buffer, size=A.size) =
    CLArray(ctx, queue, buffer, size)
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
    return CLArray(buf, dims)
end

Base.zeros{T}(::Type{T}, q::CmdQueue, dims...) = fill(T, q, T(0), dims...)
Base.zeros(q::CmdQueue, dims...) = fill(Float64, q, Float64(0), dims...)
Base.ones{T}(::Type{T}, q::CmdQueue, dims...) = fill(T, q, T(1), dims...)
Base.ones(q::CmdQueue, dims...) = fill(Float64, q, Float64(1), dims...)


##  core functions

buffer(A::CLArray) = A.buffer
bufptr(A::CLArray) = A.buffer.id
context(A::CLArray) = context(A.buffer)
Base.size(A::CLArray) = A.size
Base.ndims(A::CLArray) = length(size(A))
Base.length(A::CLArray) = prod(size(A))
Base.(:(==))(A:: CLArray, B:: CLArray) =
    buffer(A) == buffer(B) && size(A) == size(B)
Base.reshape(A::CLArray, dims...) = begin
    @assert prod(dims) == prod(size(A))
    return copy(A, size=dims)
end

##  show

Base.show{T,N}(io::IO, A::CLArray{T,N}) =
    print(io, "CLArray{$T,$N}($(buffer(A)),$(size(A)))")


##  to_host

function to_host{T,N}(A::CLArray{T,N}; queue=A.queue)
    hA = Array(T, size(A))
    copy!(queue, hA, buffer(A))
    return hA
end

## kernels

"""
Format string using dict-like variables, replacing all accurancies of
`%(key)` with `value`.

Example:
    s = "Hello, %(name)"
    format(s, name="Tom")  ==> "Hello, Tom"
"""
function format(s::AbstractString; vars...)
    for (k, v) in vars
        s = replace(s, "%($k)", v)
    end
    s
end

const PROGRAM_TRANSPOSE = readall(Pkg.dir("OpenCL/src/kernels/transpose.cl"))

function build_kernel(ctx::Context, program::AbstractString,
                      kernel_name::AbstractString; vars...)
    src = format(PROGRAM_TRANSPOSE; vars...)
    p = Program(ctx, source=src)
    build!(p)
    return Kernel(p, kernel_name)
end

## other array operations


"""Transpose CLMatrix A, write result to a preallicated CLMatrix B"""
function Base.transpose!(B::CLMatrix{Float32}, A::CLMatrix{Float32};
                         queue=A.queue, block_size=32)
    ctx = context(A)
    kernel = build_kernel(ctx, PROGRAM_TRANSPOSE, "transpose",
                          block_size=block_size)
    h, w = size(A)
    lmem = LocalMem(Float32, block_size * (block_size + 1))
    set_args!(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem)
    return enqueue_kernel(queue, kernel, (h, w), (block_size, block_size))
end

"""Transpose CLMatrix A"""
function Base.transpose(A::CLMatrix{Float32};
                        queue=A.queue, block_size=32)
    B = zeros(Float32, queue, reverse(size(A))...)
    ev = transpose!(B, A, queue=queue, block_size=block_size)
    wait(ev)
    return B
end

"""Transpose CLMatrix A, write result to a preallicated CLMatrix B"""
function Base.transpose!(B::CLMatrix{Float64}, A::CLMatrix{Float64};
                         queue=A.queue, block_size=32)
    ctx = context(A)
    kernel = build_kernel(ctx, PROGRAM_TRANSPOSE, "transpose_double",
                          block_size=block_size)
    h, w = size(A)
    lmem = LocalMem(Float64, block_size * (block_size + 1))
    set_args!(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem)
    return enqueue_kernel(queue, kernel, (h, w), (block_size, block_size))
end

"""Transpose CLMatrix A"""
function Base.transpose(A::CLMatrix{Float64};
                        queue=A.queue, block_size=32)
    B = zeros(Float64, queue, reverse(size(A))...)
    ev = transpose!(B, A, queue=queue, block_size=block_size)
    wait(ev)
    return B
end




function main()
    ## import OpenCL: CLArray, CLObject, Buffer, CmdQueue, Program, Kernel, LocalMem
    ## import OpenCL: create_compute_context, build!, context, set_args!
    ## import OpenCL: enqueue_kernel, build_kernel
    import OpenCL: CLArray
    const cl = OpenCL
    device, ctx, queue = cl.create_compute_context()
    A = CLArray(ctx, rand(Float32, 64, 64))
    B = cl.zeros(Float32, ctx, 64, 64)
    ev = transpose!(B, A, queue=queue)


end
