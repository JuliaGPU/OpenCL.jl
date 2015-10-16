
type CLArray{T,N} <: CLObject
    buffer::Buffer{T}
    size::NTuple{N,Int}
    t::Bool
end

typealias CLMatrix{T} CLArray{T,2}
typealias CLVector{T} CLArray{T,1}

##  main constructors (overload them if you add new fields)

function CLArray{T,N}(ctx::Context, flags::Tuple{Vararg{Symbol}},
                      hostarray::AbstractArray{T,N})
    buf = Buffer(T, ctx, flags, hostbuf=hostarray)
    sz = size(hostarray)
    CLArray(buf, sz, false)
end

copy(A::CLArray) = CLArray(A.buffer, A.size, A.t)
    
##  convenient constructors

CLArray{T,N}(ctx::Context, hostarray::AbstractArray{T,N}) = 
    CLArray(ctx, (:rw, :copy), hostarray)

# TODO: OpenCL may have faster equivalent
Base.zeros(t::Type, ctx::Context, dims...) = CLArray(ctx, zeros(t, dims...))
Base.zeros(ctx::Context, dims...) = CLArray(ctx, zeros(Float64, dims...))
Base.ones(t::Type, ctx::Context, dims...) = CLArray(ctx, ones(t, dims...))
Base.ones(ctx::Context, dims...) = CLArray(ctx, ones(t, dims...))


##  core functions

buffer(A::CLArray) = A.buffer
istransposed(A::CLArray) = A.t
Base.size(A::CLArray) = istransposed(A) ? reverse(A.size) : A.size
Base.ndims(A::CLArray) = length(size(A))
Base.length(A::CLArray) = prod(size(A))
Base.reshape(A::CLArray, dims...) = begin
    @assert prod(dims) == prod(size(A))
    A2 = copy(A)
    A2.size = dims
    return A2
end
Base.transpose{T}(A::CLArray{T,2}) = begin
    A2 = copy(A)
    A2.t = !A.t
    return A2
end



##  to_host

to_host{T,N}(q::CmdQueue, A::CLArray{T,N}) = begin
    hA = Array(T, size(A))
    copy!(q, hA, buffer(A))
    return istransposed(A) ? hA' : A
end


##  show

Base.show{T,N}(io::IO, A::CLArray{T,N}) = 
    print(io, "CLArray{$T,$N}($(buffer(A)),$(size(A)))")

