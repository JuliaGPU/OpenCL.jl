export CLArray, CLVector, CLMatrix, CLVecOrMat, is_device, is_shared, is_host


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

mutable struct CLArray{T,N,M} <: AbstractGPUArray{T,N}
  data::DataRef{Managed{M}}

  maxsize::Int  # maximum data size; excluding any selector bytes
  offset::Int   # offset of the data in memory, in number of elements

  dims::Dims{N}

  function CLArray{T,N,M}(::UndefInitializer, dims::Dims{N}) where {T,N,M}
    check_eltype(T)
    maxsize = prod(dims) * sizeof(T)
    bufsize = if Base.isbitsunion(T)
      # type tag array past the data
      maxsize + prod(dims)
    else
      maxsize
    end

    GPUArrays.cached_alloc((CLArray, cl.device(), T, bufsize, M)) do
        data = DataRef(managed -> release(managed.mem), Managed(allocate(M, cl.context(), cl.device(), bufsize, Base.datatype_alignment(T))))
        obj = new{T,N,M}(data, maxsize, 0, dims)
        finalizer(unsafe_free!, obj)
        return obj
    end::CLArray{T, N, M}
  end

  function CLArray{T,N}(data::DataRef{Managed{M}}, dims::Dims{N};
                        maxsize::Int=prod(dims) * sizeof(T), offset::Int=0) where {T,N,M}
    check_eltype(T)
    obj = new{T,N,M}(data, maxsize, offset, dims)
    finalizer(unsafe_free!, obj)
  end
end

GPUArrays.storage(a::CLArray) = a.data


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
CLArray{T,N,M}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,M} =
  CLArray{T,N,M}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T,N,M}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,M} =
  CLArray{T,N,M}(undef, convert(Tuple{Vararg{Int}}, dims))

# type and dimensionality specified
CLArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
CLArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  CLArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
CLArray{T,1,M}() where {T,M} = CLArray{T,1,M}(undef, 0)
CLArray{T,1}() where {T} = CLArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:CLArray => (),
                      :(CLArray{T}) => (:T,),
                      :(CLArray{T,N}) => (:T, :N),
                      :(CLArray{T,N,M}) => (:T, :N, :M))
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

Base.similar(a::CLArray{T,N,M}) where {T,N,M} =
  CLArray{T,N,M}(undef, size(a))
Base.similar(a::CLArray{T,<:Any,M}, dims::Base.Dims{N}) where {T,N,M} =
  CLArray{T,N,M}(undef, dims)
Base.similar(a::CLArray{<:Any,<:Any,M}, ::Type{T}, dims::Base.Dims{N}) where {T,N,M} =
  CLArray{T,N,M}(undef, dims)

function Base.copy(a::CLArray{T,N}) where {T,N}
  b = similar(a)
  @inbounds copyto!(b, a)
end

function Base.deepcopy_internal(x::CLArray, dict::IdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end


## unsafe_wrap

"""
  # simple case, wrapping a CLArray around an existing GPU pointer
  unsafe_wrap(CLArray, ptr::CLPtr{T}, dims; own=false, ctx=context())

  # wraps a CPU array object around a unified GPU array
  unsafe_wrap(Array, a::CLArray)

  # wraps a GPU array object around a CPU array.
  # if your system supports Unified Memory, this is a fast operation.
  # in other cases, it has to use page locking, which can be slow.
  unsafe_wrap(CLArray, ptr::ptr{T}, dims)
  unsafe_wrap(CLArray, a::Array)

Wrap a `CLArray` object around the data at the address given by the cl-managed pointer
`ptr`. The element type `T` determines the array element type. `dims` is either an integer
(for a 1d array) or a tuple of the array dimensions. `own` optionally specified whether
Julia should take ownership of the memory, calling `cudaFree` when the array is no longer
referenced. The `ctx` argument determines the cl context where the data is allocated in.
"""
unsafe_wrap

#= TODO: Look into managed pointer memory in OpenCL
# managed pointer to CLArray
function Base.unsafe_wrap(::Union{Type{CLArray},Type{CLArray{T}},Type{CLArray{T,N}}},
                          ptr::CLPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::cl.Context=cl.context()) where {T,N}
  # identify the memory type
  M = try
    typ = buftype(ptr)
    if is_managed(ptr)
      cl.ShareddBuffer
    elseif typ == CU_MEMORYTYPE_DEVICE
      cl.DeviceBuffer
    elseif typ == CU_MEMORYTYPE_HOST
      cl.HostBuffer
    else
      error("Unknown memory type; please file an issue.")
    end
  catch err
      throw(ArgumentError("Could not identify the memory type; are you passing a valid cl pointer to unsafe_wrap?"))
  end

  unsafe_wrap(CLArray{T,N,M}, ptr, dims; own, ctx)
end

function Base.unsafe_wrap(::Type{CLArray{T,N,M}},
                          ptr::CLPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::cl.Context=cl.context()) where {T,N,M}
  isbitstype(T) || throw(ArgumentError("Can only unsafe_wrap a pointer to a bits type"))
  sz = prod(dims) * sizeof(T)

  # create a memory object
  mem = if M == cl.SharedBuffer
    cl.SharedBuffer(ctx, ptr, sz)
  elseif M == cl.DeviceBuffer
    # TODO: can we identify whether this pointer was allocated asynchronously?
    cl.DeviceBuffer(device(ctx), ctx, ptr, sz, false)
  elseif M == cl.HostBuffer
    cl.HostBuffer(ctx, host_pointer(ptr), sz)
  else
    throw(ArgumentError("Unknown memory type $M"))
  end

  data = DataRef(own ? pool_free : Returns(nothing), Managed(mem))
  CLArray{T,N}(data, dims)
end
# integer size input
function Base.unsafe_wrap(::Union{Type{CLArray},Type{CLArray{T}},Type{CLArray{T,1}}},
                          p::CLPtr{T}, dim::Int;
                          own::Bool=false, ctx::CLContext=context()) where {T}
  unsafe_wrap(CLArray{T,1}, p, (dim,); own, ctx)
end
function Base.unsafe_wrap(::Type{CLArray{T,1,M}}, p::CLPtr{T}, dim::Int;
                          own::Bool=false, ctx::CLContext=context()) where {T,M}
  unsafe_wrap(CLArray{T,1,M}, p, (dim,); own, ctx)
end

# managed pointer to Array
function Base.unsafe_wrap(::Union{Type{Array},Type{Array{T}},Type{Array{T,N}}},
                          p::CLPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false) where {T,N}
  if !is_managed(p) && buftype(p) != CU_MEMORYTYPE_HOST
    throw(ArgumentError("Can only create a CPU array object from a unified or host cl array"))
  end
  unsafe_wrap(Array{T,N}, reinterpret(Ptr{T}, p), dims; own)
end
# integer size input
function Base.unsafe_wrap(::Union{Type{Array},Type{Array{T}},Type{Array{T,1}}},
                          p::CLPtr{T}, dim::Int; own::Bool=false) where {T}
  unsafe_wrap(Array{T,1}, p, (dim,); own)
end
# array input
function Base.unsafe_wrap(::Union{Type{Array},Type{Array{T}},Type{Array{T,N}}},
                          a::CLArray{T,N}) where {T,N}
  p = pointer(a; type=HostBuffer)
  unsafe_wrap(Array, p, size(a))
end

# unmanaged pointer to CLArray
supports_hmm(dev) = driver_version() >= v"12.2" &&
                    attribute(dev, DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS) == 1
function Base.unsafe_wrap(::Type{CLArray{T,N,M}}, p::Ptr{T}, dims::NTuple{N,Int};
                          ctx::CLContext=context()) where {T,N,M<:AbstractBuffer}
  isbitstype(T) || throw(ArgumentError("Can only unsafe_wrap a pointer to a bits type"))
  sz = prod(dims) * sizeof(T)

  data = if M == SharedBuffer
    # HMM extends unified memory to include system memory
    supports_hmm(device(ctx)) ||
      throw(ArgumentError("Cannot wrap system memory as unified memory on your system"))
    mem = SharedBuffer(ctx, reinterpret(CLPtr{Nothing}, p), sz)
    DataRef(Returns(nothing), Managed(mem))
  elseif M == HostBuffer
    # register as device-accessible host memory
    mem = context!(ctx) do
      register(HostBuffer, p, sz, MEMHOSTREGISTER_DEVICEMAP)
    end
    DataRef(Managed(mem)) do args...
      context!(ctx; skip_destroyed=true) do
        unregister(mem)
      end
    end
  else
    throw(ArgumentError("Cannot wrap system memory as $M"))
  end

  CLArray{T,N}(data, dims)
end
function Base.unsafe_wrap(::Union{Type{CLArray},Type{CLArray{T}},Type{CLArray{T,N}}},
                          p::Ptr{T}, dims::NTuple{N,Int}; ctx::CLContext=context()) where {T,N}
  if supports_hmm(device(ctx))
    Base.unsafe_wrap(CLArray{T,N,SharedBuffer}, p, dims; ctx)
  else
    Base.unsafe_wrap(CLArray{T,N,HostBuffer}, p, dims; ctx)
  end
end
# integer size input
Base.unsafe_wrap(::Union{Type{CLArray},Type{CLArray{T}},Type{CLArray{T,1}}},
                 p::Ptr{T}, dim::Int) where {T} =
  unsafe_wrap(CLArray{T,1}, p, (dim,))
Base.unsafe_wrap(::Type{CLArray{T,1,M}}, p::Ptr{T}, dim::Int) where {T,M} =
  unsafe_wrap(CLArray{T,1,M}, p, (dim,))
# array input
Base.unsafe_wrap(::Union{Type{CLArray},Type{CLArray{T}},Type{CLArray{T,N}}},
                 a::Array{T,N}) where {T,N} =
  unsafe_wrap(CLArray{T,N}, pointer(a), size(a))
Base.unsafe_wrap(::Type{CLArray{T,N,M}}, a::Array{T,N}) where {T,N,M} =
  unsafe_wrap(CLArray{T,N,M}, pointer(a), size(a))
=#

## array interface

Base.elsize(::Type{<:CLArray{T}}) where {T} = sizeof(T)

Base.size(x::CLArray) = x.dims
Base.sizeof(x::CLArray) = Base.elsize(x) * length(x)

context(A::CLArray) = cl.context(A.data[].mem)
device(A::CLArray) = cl.device(A.data[].mem)

buftype(x::CLArray) = buftype(typeof(x))
buftype(::Type{<:CLArray{<:Any,<:Any,M}}) where {M} = @isdefined(M) ? M : Any

is_device(a::CLArray) = buftype(a) == cl.DeviceBuffer
is_shared(a::CLArray) = buftype(a) == cl.SharedBuffer
is_host(a::CLArray) = buftype(a) == cl.HostBuffer


## derived types

export DenseCLArray, DenseCLVector, DenseCLMatrix, DenseCLVecOrMat,
       StridedCLArray, StridedCLVector, StridedCLMatrix, StridedCLVecOrMat,
       WrappedCLArray, WrappedCLVector, WrappedCLMatrix, WrappedCLVecOrMat

# dense arrays: stored contiguously in memory
#
# all common dense wrappers are currently represented as CLArray objects.
# this simplifies common use cases, and greatly improves load time.
# cl.jl 2.0 experimented with using ReshapedArray/ReinterpretArray/SubArray,
# but that proved much too costly. TODO: revisit when we have better Base support.
const DenseCLArray{T,N} = CLArray{T,N}
const DenseCLVector{T} = DenseCLArray{T,1}
const DenseCLMatrix{T} = DenseCLArray{T,2}
const DenseCLVecOrMat{T} = Union{DenseCLVector{T}, DenseCLMatrix{T}}
# XXX: these dummy aliases (DenseCLArray=CLArray) break alias printing, as
#      `Base.print_without_params` only handles the case of a single alias.

# strided arrays
const StridedSubCLArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                            Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:CLArray,I}
const StridedCLArray{T,N} = Union{CLArray{T,N}, StridedSubCLArray{T,N}}
const StridedCLVector{T} = StridedCLArray{T,1}
const StridedCLMatrix{T} = StridedCLArray{T,2}
const StridedCLVecOrMat{T} = Union{StridedCLVector{T}, StridedCLMatrix{T}}

@inline function Base.pointer(x::StridedCLArray{T}, i::Integer=1; type=cl.DeviceBuffer) where T
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
const WrappedCLArray{T,N} = Union{CLArray{T,N}, WrappedArray{T,N,CLArray,CLArray{T,N}}}
const WrappedCLVector{T} = WrappedCLArray{T,1}
const WrappedCLMatrix{T} = WrappedCLArray{T,2}
const WrappedCLVecOrMat{T} = Union{WrappedCLVector{T}, WrappedCLMatrix{T}}


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
#=
# defer the conversion to Managed, where we handle memory consistency
# XXX: conversion to Buffer or Managed memory by cconvert?
Base.unsafe_convert(typ::Type{Ptr{T}}, x::CLArray{T}) where {T} =
  convert(typ, x.data[]) + x.offset * Base.elsize(x)
Base.unsafe_convert(typ::Type{CLPtr{T}}, x::CLArray{T}) where {T} =
  convert(typ, x.data[]) + x.offset * Base.elsize(x)
=#

## indexing

function Base.getindex(x::CLArray{<:Any, <:Any, <:Union{cl.HostBuffer,cl.SharedBuffer}}, I::Int)
  @boundscheck checkbounds(x, I)
  unsafe_load(pointer(x, I; type=cl.HostBuffer))
end

function Base.setindex!(x::CLArray{<:Any, <:Any, <:Union{cl.HostBuffer,cl.SharedBuffer}}, v, I::Int)
  @boundscheck checkbounds(x, I)
  unsafe_store!(pointer(x, I; type=cl.HostBuffer), v)
end

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

## interop with device arrays

function Base.unsafe_convert(::Type{CLDeviceArray{T,N,AS.Global}}, a::DenseCLArray{T,N}) where {T,N}
  CLDeviceArray{T,N,AS.Global}(reinterpret(LLVMPtr{T,AS.Global}, pointer(a)), size(a),
                               a.maxsize - a.offset*Base.elsize(a))
end


## memory copying

synchronize(x::CLArray) = synchronize(x.data[])

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

Base.copyto!(dest::DenseCLArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::Array{T}, doffs::Integer, src::DenseCLArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(context(src), cl.device(), dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::Array{T}, src::DenseCLArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::DenseCLArray{T}, doffs::Integer, src::DenseCLArray{T}, soffs::Integer,
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

Base.copyto!(dest::DenseCLArray{T}, src::DenseCLArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

for (srcty, dstty) in [(:Array, :CLArray), (:CLArray, :Array), (:CLArray, :CLArray)]
    @eval begin
        function Base.unsafe_copyto!(dst::$dstty{T}, dst_off::Int,
                                     src::$srcty{T}, src_off::Int,
                                     N::Int; blocking::Bool=true) where T
            nbytes = N * sizeof(T)
            cl.enqueue_usm_memcpy(pointer(dst, dst_off), pointer(src, src_off), nbytes;
                                  blocking)
        end
        Base.unsafe_copyto!(dst::$dstty, src::$srcty, N; kwargs...) =
            unsafe_copyto!(dst, 1, src, 1, N; kwargs...)
    end
end

function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device,
                             dest::DenseCLArray{T}, doffs, src::Array{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("CLArray does not yet support isbits-union arrays")
  end
  return dest
end

function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device,
                             dest::Array{T}, doffs, src::DenseCLArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("CLArray does not yet support isbits-union arrays")
  end

  # copies to the host are synchronizing
  synchronize(src)

  return dest
end

function Base.unsafe_copyto!(ctx::cl.Context, dev::cl.Device,
                             dest::DenseCLArray{T}, doffs, src::DenseCLArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(ctx, dev, pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("CLArray does not yet support isbits-union arrays")
  end
  return dest
end

# between Array and host-accessible CLArray

function Base.unsafe_copyto!(ctx::cl.cl.Context, dev::cl.Device,
                             dest::DenseCLArray{T,<:Any,<:Union{cl.SharedBuffer,cl.HostBuffer}}, doffs, src::Array{T}, soffs, n) where T
  # maintain queue-ordered semantics
  synchronize(dest)

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
                             dest::Array{T}, doffs, src::DenseCLArray{T,<:Any,<:Union{cl.SharedBuffer,cl.HostBuffer}}, soffs, n) where T
  # maintain queue-ordered semantics
  synchronize(src)

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

#= TODO: LOOK INTO IF THIS OPTIMIZATION CAN BE SUPPORTED
# optimization: memcpy between host or unified arrays without context switching

function Base.unsafe_copyto!(dest::DenseCLArray{T,<:Any,<:Union{cl.SharedBuffer,cl.HostBuffer}}, doffs,
                             src::DenseCLArray{T}, soffs, n) where T
  context!(context(src)) do
    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::DenseCLArray{T}, doffs,
                             src::DenseCLArray{T,<:Any,<:Union{cl.SharedBuffer,cl.HostBuffer}}, soffs, n) where T
  context!(context(dest)) do
    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::DenseCLArray{T,<:Any,<:Union{SharedBuffer,HostBuffer}}, doffs,
                             src::DenseCLArray{T,<:Any,<:Union{cl.SharedBuffer,cl.HostBuffer}}, soffs, n) where T
  GC.@preserve src dest begin
    unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
    if Base.isbitsunion(T)
      unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
    end
  end
  return dest
end
=#

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
Adapt.adapt_storage(::Type{<:CLArray{T, N, M}}, xs::AT) where {T, N, M, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CLArray{T,N,M}, xs)

#= TODO: LOOK INTO IF THIS IS OKAY OR NOT, LATER
## opinionated gpu array adaptor

# eagerly converts Float64 to Float32, for performance reasons

struct CLArrayKernelAdaptor{M} end

Adapt.adapt_storage(::CLArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T,N,M} =
  isbits(xs) ? xs : CLArray{T,N,M}(xs)

Adapt.adapt_storage(::CLArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:AbstractFloat,N,M} =
  isbits(xs) ? xs : CLArray{Float32,N,M}(xs)

Adapt.adapt_storage(::CLArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:Complex{<:AbstractFloat},N,M} =
  isbits(xs) ? xs : CLArray{ComplexF32,N,M}(xs)

# not for Float16
Adapt.adapt_storage(::CLArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:Union{Float16,BFloat16},N,M} =
  isbits(xs) ? xs : CLArray{T,N,M}(xs)
=#

## utilities

zeros(T::Type, dims...) = fill!(CLArray{T}(undef, dims...), zero(T))
ones(T::Type, dims...) = fill!(CLArray{T}(undef, dims...), one(T))
zeros(dims...) = zeros(Float32, dims...)
ones(dims...) = ones(Float32, dims...)
fill(v, dims...) = fill!(CLArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(CLArray{typeof(v)}(undef, dims...), v)

#= TODO: Look into this optimization later
# optimized implementation of `fill!` for types that are directly supported by memset
memsettype(T::Type) = T
memsettype(T::Type{<:Signed}) = unsigned(T)
memsettype(T::Type{<:AbstractFloat}) = Base.uinttype(T)
const MemsetCompatTypes = Union{UInt8, Int8,
                                UInt16, Int16, Float16,
                                UInt32, Int32, Float32}
function Base.fill!(A::DenseCLArray{T}, x) where T <: MemsetCompatTypes
  U = memsettype(T)
  y = reinterpret(U, convert(T, x))
  context!(context(A)) do
    memset(convert(CLPtr{U}, pointer(A)), y, length(A))
  end
  A
end
=#

function Base.fill!(A::DenseCLArray{T}, val) where T
  B = [convert(T, val)]
  unsafe_fill!(context(A), cl.device(), pointer(A), pointer(B), length(A))
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
