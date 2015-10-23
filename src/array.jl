
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

function CLArray{T}(buf::Buffer{T}, sz::Tuple{Vararg{Int}})
    ctx = context(buf)
    queue = CmdQueue(ctx)
    CLArray(context(buf), queue, buf, sz)
end
    
Base.copy(A::CLArray) = CLArray(A.ctx, A.queue, A.buffer, A.size)

# TODO: OpenCL may have faster equivalent
## Base.zeros(t::Type, ctx::Context, dims...) = CLArray(ctx, zeros(t, dims...))
## Base.zeros(ctx::Context, dims...) = CLArray(ctx, zeros(Float64, dims...))
## Base.ones(t::Type, ctx::Context, dims...) = CLArray(ctx, ones(t, dims...))
## Base.ones(ctx::Context, dims...) = CLArray(ctx, ones(Float64, dims...))


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
    A2 = copy(A)
    A2.size = dims
    return A2
end

##  show

Base.show{T,N}(io::IO, A::CLArray{T,N}) =
    print(io, "CLArray{$T,$N}($(buffer(A)),$(size(A)))")


##  to_host

function to_host{T,N}(q::CmdQueue, A::CLArray{T,N})
    hA = Array(T, size(A))
    copy!(q, hA, buffer(A))
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

const PROGRAM_TRANSPOSE = readall("kernels/transpose.cl")

function build_kernel(ctx::Context, program::AbstractString,
                      kernel_name::AbstractString; vars...)
    src = format(PROGRAM_TRANSPOSE; vars...)
    p = Program(ctx, source=src)
    build!(p)
    return Kernel(p, kernel_name)    
end

## other array operations


"""
Transpose CLMatrix A, write result to a preallicated CLMatrix B
"""
function transpose!(B::CLMatrix{Float32}, A::CLMatrix{Float32};
                        queue::Union{CmdQueue, Void}=nothing, block_size=32)
    ctx = context(A)
    queue = queue == nothing ? CmdQueue(ctx) : queue
    kernel = build_kernel(ctx, PROGRAM_TRANSPOSE, "transpose",
                          block_size=block_size)
    h, w = size(A)
    lmem = LocalMem(Float32, block_size * (block_size + 1))
    set_args!(kernel, buffer(B), buffer(A), UInt32(h), UInt32(w), lmem)
    return enqueue_kernel(queue, kernel, (h, w), (block_size, block_size))    
end

## Base.transpose(A::CLMatrix{Float64};
##                queue::Union{CmdQueue, Void}=nothing, block_size=32) =
##                    transpose!(zeros(Float32, context(A), ))





function main()
    import OpenCL: CLArray, CLObject, Buffer, CmdQueue, Program, Kernel, LocalMem
    import OpenCL: create_compute_context, build!, context, set_args!
    import OpenCL: enqueue_kernel, build_kernel
    const cl = OpenCL
    device, ctx, queue = cl.create_compute_context()
    A = CLArray(ctx, rand(Float32, 64, 64))
    B = cl.zeros(Float32, ctx, 64, 64)
    ev = transpose!(B, A, queue=queue)
    

end

