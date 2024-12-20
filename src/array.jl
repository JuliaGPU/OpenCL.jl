export CLArray, CLVector, CLMatrix, CLVecOrMat,
       is_device, is_shared, is_host


## array type

function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

function contains_eltype(T, X)
    if T === X
      return true
    elseif T isa Union
        for U in Base.uniontypes(T)
            contains_eltype(U, X) && return true
        end
    elseif hasfieldcount(T)
        for U in fieldtypes(T)
            contains_eltype(U, X) && return true
        end
    end
    return false
end

function check_eltype(T)
  Base.allocatedinline(T) || error("CLArray only supports element types that are stored inline")
  Base.isbitsunion(T) && error("CLArray does not yet support isbits-union arrays")
  !("cl_khr_fp16" in cl.device().extensions) && contains_eltype(T, Float16) && error("Float16 is not supported on this device")
  !("cl_khr_fp64" in cl.device().extensions) && contains_eltype(T, Float64) && error("Float16 is not supported on this device")
end

mutable struct CLArray{T,N,B} <: AbstractGPUArray{T,N}
  data::DataRef{B}

  maxsize::Int  # maximum data size; excluding any selector bytes
  offset::Int   # offset of the data in the buffer, in number of elements
  dims::Dims{N}

  function CLArray{T,N,B}(::UndefInitializer, dims::Dims{N}) where {T,N,B}
    check_eltype(T)
    maxsize = prod(dims) * sizeof(T)
    bufsize = if Base.isbitsunion(T)
      # type tag array past the data
      maxsize + prod(dims)
    else
      maxsize
    end

    ctx = cl.context()
    dev = cl.device()
    buf = cl.allocate(B, ctx, dev, bufsize, Base.datatype_alignment(T))
    data = DataRef(buf) do buf
      release(buf)
    end
    obj = new{T,N,B}(data, maxsize, 0, dims)
    finalizer(unsafe_free!, obj)
  end

  function CLArray{T,N}(data::DataRef{B}, dims::Dims{N};
                         maxsize::Int=prod(dims) * sizeof(T), offset::Int=0) where {T,N,B}
    check_eltype(T)
    if sizeof(T) == 0
      offset == 0 || error("Singleton arrays cannot have a nonzero offset")
      maxsize == 0 || error("Singleton arrays cannot have a size")
    end
    obj = new{T,N,B}(copy(data), maxsize, offset, dims)
    finalizer(unsafe_free!, obj)
  end
end

unsafe_free!(a::CLArray) = GPUArrays.unsafe_free!(a.data)


## alias detection

Base.dataids(A::CLArray) = (UInt(pointer(A)),)

Base.unaliascopy(A::CLArray) = copy(A)

function Base.mightalias(A::CLArray, B::CLArray)
  rA = pointer(A):pointer(A)+sizeof(A)
  rB = pointer(B):pointer(B)+sizeof(B)
  return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

const CLVector{T} = CLArray{T,1}
const CLMatrix{T} = CLArray{T,2}
const CLVecOrMat{T} = Union{CLVector{T},CLMatrix{T}}

# default to non-unified memory
CLArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  CLArray{T,N,cl.DeviceBuffer}(undef, dims)

# buffer, type and dimensionality specified
CLArray{T,N,B}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,B} =
  CLArray{T,N,B}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T,N,B}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,B} =
  CLArray{T,N,B}(undef, convert(Tuple{Vararg{Int}}, dims))

# type and dimensionality specified
CLArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# only type specified
CLArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
CLArray{T,1,B}() where {T,B} = CLArray{T,1,B}(undef, 0)
CLArray{T,1}() where {T} = CLArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:CLArray => (),
                      :(CLArray{T}) => (:T,),
                      :(CLArray{T,N}) => (:T, :N),
                      :(CLArray{T,N,B}) => (:T, :N, :B))
  @eval begin
    function $ctor(f::Function, args...) where {$(tvars...)}
      xs = $ctor(args...)
      try
        f(xs)
      finally
        unsafe_free!(xs)
      end
    end
  end
end

Base.similar(a::CLArray{T,N,B}) where {T,N,B} =
  CLArray{T,N,B}(undef, size(a))
Base.similar(a::CLArray{T,<:Any,B}, dims::Base.Dims{N}) where {T,N,B} =
  CLArray{T,N,B}(undef, dims)
Base.similar(a::CLArray{<:Any,<:Any,B}, ::Type{T}, dims::Base.Dims{N}) where {T,N,B} =
  CLArray{T,N,B}(undef, dims)

function Base.copy(a::CLArray{T,N}) where {T,N}
  b = similar(a)
  @inbounds copyto!(b, a)
end


## array interface

Base.elsize(::Type{<:CLArray{T}}) where {T} = sizeof(T)

Base.size(x::CLArray) = x.dims
Base.sizeof(x::CLArray) = Base.elsize(x) * length(x)

function context(A::CLArray)
  return cl.context(A.data[])
end

function device(A::CLArray)
  return cl.device(A.data[])
end

buftype(x::CLArray) = buftype(typeof(x))
buftype(::Type{<:CLArray{<:Any,<:Any,B}}) where {B} = @isdefined(B) ? B : Any

is_device(a::CLArray) = isa(a.data[], cl.DeviceBuffer)
is_shared(a::CLArray) = isa(a.data[], cl.SharedBuffer)
is_host(a::CLArray) = isa(a.data[], cl.HostBuffer)

## derived types

export CLDenseArray, CLDenseVector, CLDenseMatrix, CLDenseVecOrMat,
       CLStridedArray, CLStridedVector, CLStridedMatrix, CLStridedVecOrMat,
       CLWrappedArray, CLWrappedVector, CLWrappedMatrix, CLWrappedVecOrMat

# dense arrays: stored contiguously in memory
#
# all common dense wrappers are currently represented as CLArray objects.
# this simplifies common use cases, and greatly improves load time.
const CLDenseArray{T,N} = CLArray{T,N}
const CLDenseVector{T} = CLDenseArray{T,1}
const CLDenseMatrix{T} = CLDenseArray{T,2}
const CLDenseVecOrMat{T} = Union{CLDenseVector{T}, CLDenseMatrix{T}}
# XXX: these dummy aliases (CLDenseArray=CLArray) break alias printing, as
#      `Base.print_without_params` only handles the case of a single alias.

# strided arrays
const CLStridedSubArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                             Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:CLArray,I}
const CLStridedArray{T,N} = Union{CLArray{T,N}, CLStridedSubArray{T,N}}
const CLStridedVector{T} = CLStridedArray{T,1}
const CLStridedMatrix{T} = CLStridedArray{T,2}
const CLStridedVecOrMat{T} = Union{CLStridedVector{T}, CLStridedMatrix{T}}

@inline function Base.pointer(x::CLStridedArray{T}, i::Integer=1; type=cl.DeviceBuffer) where T
    PT = if type == cl.DeviceBuffer
      CLPtr{T}
    elseif type == cl.HostBuffer
      Ptr{T}
    else
      error("unknown memory type")
    end
    Base.unsafe_convert(PT, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a CLArray
const CLWrappedArray{T,N} = Union{CLArray{T,N}, WrappedArray{T,N,CLArray,CLArray{T,N}}}
const CLWrappedVector{T} = CLWrappedArray{T,1}
const CLWrappedMatrix{T} = CLWrappedArray{T,2}
const CLWrappedVecOrMat{T} = Union{CLWrappedVector{T}, CLWrappedMatrix{T}}


## interop with other arrays

@inline function CLArray{T,N,B}(xs::AbstractArray{<:Any,N}) where {T,N,B}
  A = CLArray{T,N,B}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

@inline CLArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N} =
  CLArray{T,N,cl.DeviceBuffer}(xs)

@inline CLArray{T,N}(xs::CLArray{<:Any,N,B}) where {T,N,B} =
  CLArray{T,N,B}(xs)

# underspecified constructors
CLArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = CLArray{T,N}(xs)
(::Type{CLArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = CLArray{S,N}(x)
CLArray(A::AbstractArray{T,N}) where {T,N} = CLArray{T,N}(A)

# idempotency
CLArray{T,N,B}(xs::CLArray{T,N,B}) where {T,N,B} = xs
CLArray{T,N}(xs::CLArray{T,N,B}) where {T,N,B} = xs

# Level CLro references
cl.CLRef(x::Any) = cl.CLRefArray(CLArray([x]))
cl.CLRef{T}(x) where {T} = cl.CLRefArray{T}(CLArray(T[x]))
cl.CLRef{T}() where {T} = cl.CLRefArray(CLArray{T}(undef, 1))


## conversions

Base.convert(::Type{T}, x::T) where T <: CLArray = x


## interop with libraries

function Base.unsafe_convert(::Type{Ptr{T}}, x::CLArray{T}) where {T}
  buf = x.data[]
  if is_device(x)
    throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
  end
  convert(Ptr{T}, x.data[]) + x.offset*Base.elsize(x)
end

function Base.unsafe_convert(::Type{CLPtr{T}}, x::CLArray{T}) where {T}
  convert(CLPtr{T}, x.data[]) + x.offset*Base.elsize(x)
end



## interop with GPU arrays

function Base.unsafe_convert(::Type{CLDeviceArray{T,N,AS.Global}}, a::CLArray{T,N}) where {T,N}
  CLDeviceArray{T,N,AS.Global}(size(a), reinterpret(LLVMPtr{T,AS.Global}, pointer(a)),
                                a.maxsize - a.offset*Base.elsize(a))
end


## memory copying

typetagdata(a::Array, i=1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
typetagdata(a::CLArray, i=1) =
  convert(CLPtr{UInt8}, a.data[]) + a.maxsize + a.offset + i - 1

function Base.copyto!(dest::CLArray{T}, doffs::Integer, src::Array{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(context(dest), cl.device(), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::CLDenseArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::Array{T}, doffs::Integer, src::CLDenseArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(context(src), cl.device(), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::Array{T}, src::CLDenseArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::CLDenseArray{T}, doffs::Integer, src::CLDenseArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  @assert context(dest) == context(src)
  unsafe_copyto!(context(dest), cl.device(), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::CLDenseArray{T}, src::CLDenseArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.CLDevice,
                             dest::CLDenseArray{T}, doffs, src::Array{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("CLArray does not yet support isbits-union arrays")
  end
  return dest
end

function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device,
                             dest::Array{T}, doffs, src::CLDenseArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("CLArray does not yet support isbits-union arrays")
  end

  # copies to the host are synchronizing
  synchronize(global_queue(context(src), cl.device()))

  return dest
end

function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device,
                             dest::CLDenseArray{T}, doffs, src::CLDenseArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("CLArray does not yet support isbits-union arrays")
  end
  return dest
end

# between Array and host-accessible CLArray

function Base.unsafe_copyto!(ctx::cl.cl.Context, dev::cl.Device,
                             dest::CLDenseArray{T,<:Any,<:Union{cl.SharedBuffer,cl.HostBuffer}}, doffs, src::Array{T}, soffs, n) where T
  # maintain queue-ordered semantics
  synchronize(global_queue(ctx, dev))

  if Base.isbitsunion(T)
    # copy selector bytes
    error("CLArray does not yet support isbits-union arrays")
  end
  GC.@preserve src dest begin
    ptr = pointer(dest, doffs)
    unsafe_copyto!(pointer(dest, doffs; type=cl.HostBuffer), pointer(src, soffs), n)
    if Base.isbitsunion(T)
      # copy selector bytes
      error("CLArray does not yet support isbits-union arrays")
    end
  end

  return dest
end

function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device,
                             dest::Array{T}, doffs, src::CLDenseArray{T,<:Any,<:Union{cl.SharedBuffer,cl.HostBuffer}}, soffs, n) where T
  # maintain queue-ordered semantics
  synchronize(global_queue(ctx, dev))

  if Base.isbitsunion(T)
    # copy selector bytes
    error("CLArray does not yet support isbits-union arrays")
  end
  GC.@preserve src dest begin
    ptr = pointer(dest, doffs)
    unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs; type=cl.HostBuffer), n)
    if Base.isbitsunion(T)
      # copy selector bytes
      error("CLArray does not yet support isbits-union arrays")
    end
  end

  return dest
end


## gpu array adaptor

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CLArray}, xs::AT) where {AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CLArray, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:CLArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CLArray{T}, xs)


## utilities

zeros(T::Type, dims...) = fill!(CLArray{T}(undef, dims...), zero(T))
CLs(T::Type, dims...) = fill!(CLArray{T}(undef, dims...), CL(T))
zeros(dims...) = zeros(Float64, dims...)
CLs(dims...) = CLs(Float64, dims...)
fill(v, dims...) = fill!(CLArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(CLArray{typeof(v)}(undef, dims...), v)

function Base.fill!(A::CLDenseArray{T}, val) where T
  B = [convert(T, val)]
  unsafe_fill!(context(A), device(), pointer(A), pointer(B), length(A))
  A
end


## derived arrays

function GPUArrays.derive(::Type{T}, a::CLArray, dims::Dims{N}, offset::Int) where {T,N}
  offset = if sizeof(T) == 0
    Base.elsize(a) == 0 || error("Cannot derive a singleton array from non-singleton inputs")
    offset
  else
    (a.offset * Base.elsize(a)) ÷ sizeof(T) + offset
  end
  CLArray{T,N}(a.data, dims; a.maxsize, offset)
end


## views

device(a::SubArray) = device(parent(a))
context(a::SubArray) = context(parent(a))

# pointer conversions
function Base.unsafe_convert(::Type{CLPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P}
    return Base.unsafe_convert(CLPtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end
function Base.unsafe_convert(::Type{CLPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}}}) where {T,N,P}
   return Base.unsafe_convert(CLPtr{T}, parent(V)) +
          (Base.first_index(V)-1)*sizeof(T)
end


## PermutedDimsArray

device(a::Base.PermutedDimsArray) = device(parent(a))
context(a::Base.PermutedDimsArray) = context(parent(a))

Base.unsafe_convert(::Type{CLPtr{T}}, A::PermutedDimsArray) where {T} =
    Base.unsafe_convert(CLPtr{T}, parent(A))


## unsafe_wrap

"""
    unsafe_wrap(Array, arr::CLArray{_,_,cl.SharedBuffer})

Wrap a Julia `Array` around the buffer that backs a `CLArray`. This is only possible if the
GPU array is backed by a shared buffer, i.e. if it was created with `CLArray{T}(undef, ...)`.
"""
function Base.unsafe_wrap(::Type{Array}, arr::CLArray{T,N,cl.SharedBuffer}) where {T,N}
  # TODO: can we make this more convenient by increasing the buffer's refcount and using
  #       a finalizer on the Array? does that work when taking views etc of the Array?
  ptr = reinterpret(Ptr{T}, pointer(arr))
  unsafe_wrap(Array, ptr, size(arr))
end


## resizing

"""
  resize!(a::CLVector, n::Integer)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.
"""
function Base.resize!(a::CLVector{T}, n::Integer) where {T}
    # TODO: add additional space to allow for quicker resizing
    maxsize = n * sizeof(T)
    bufsize = if isbitstype(T)
        maxsize
    else
        # type tag array past the data
        maxsize + n
    end

    # replace the data with a new CL. this 'unshares' the array.
    # as a result, we can safely support resizing unowned buffers.
    ctx = context(a)
    dev = device(a)
    buf = allocate(buftype(a), ctx, dev, bufsize, Base.datatype_alignment(T))
    ptr = convert(CLPtr{T}, buf)
    m = min(length(a), n)
    if m > 0
        unsafe_copyto!(ctx, dev, ptr, pointer(a), m)
    end
    new_data = DataRef(buf) do buf
        free(buf)
    end
    unsafe_free!(a)

    a.data = new_data
    a.dims = (n,)
    a.maxsize = maxsize
    a.offset = 0

    a
end
