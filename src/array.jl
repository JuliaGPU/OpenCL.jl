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
    return !("cl_khr_fp64" in cl.device().extensions) && contains_eltype(T, Float64) && error("Float16 is not supported on this device")
end

mutable struct CLArray{T, N, M} <: AbstractGPUArray{T, N}
    data::DataRef{Managed{M}}

    maxsize::Int  # maximum data size; excluding any selector bytes
    offset::Int   # offset of the data in memory, in number of elements

    dims::Dims{N}

    function CLArray{T, N, M}(::UndefInitializer, dims::Dims{N}) where {T, N, M}
        check_eltype(T)
        maxsize = prod(dims) * sizeof(T)
        bufsize = if Base.isbitsunion(T)
            # type tag array past the data
            maxsize + prod(dims)
        else
            maxsize
        end
        data = GPUArrays.cached_alloc((CLArray, cl.context(), M, bufsize)) do
            buf = alloc(M, bufsize; alignment=Base.datatype_alignment(T))
            DataRef(free, buf)
        end
        obj = new{T, N, M}(data, maxsize, 0, dims)
        finalizer(unsafe_free!, obj)
        return obj
    end

    function CLArray{T, N}(
            data::DataRef{Managed{M}}, dims::Dims{N};
            maxsize::Int = prod(dims) * sizeof(T), offset::Int = 0
        ) where {T, N, M}
        check_eltype(T)
        obj = new{T, N, M}(data, maxsize, offset, dims)
        return finalizer(unsafe_free!, obj)
    end
end

GPUArrays.storage(a::CLArray) = a.data


## alias detection

Base.dataids(A::CLArray) = (UInt(pointer(A)),)

Base.unaliascopy(A::CLArray) = copy(A)

function Base.mightalias(A::CLArray, B::CLArray)
    rA = pointer(A):(pointer(A) + sizeof(A))
    rB = pointer(B):(pointer(B) + sizeof(B))
    return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

const CLVector{T} = CLArray{T, 1}
const CLMatrix{T} = CLArray{T, 2}
const CLVecOrMat{T} = Union{CLVector{T}, CLMatrix{T}}

# default to non-unified memory
function memory_type()
    if cl.memory_backend() == cl.USMBackend()
        return cl.UnifiedDeviceMemory
    elseif cl.memory_backend() == cl.SVMBackend()
        return cl.SharedVirtualMemory
    else
        return cl.BufferDeviceMemory
    end
end
CLArray{T, N}(::UndefInitializer, dims::Dims{N}) where {T, N} =
    CLArray{T, N, memory_type()}(undef, dims)

# buffer, type and dimensionality specified
CLArray{T, N, M}(::UndefInitializer, dims::NTuple{N, Integer}) where {T, N, M} =
    CLArray{T, N, M}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T, N, M}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N, M} =
    CLArray{T, N, M}(undef, convert(Tuple{Vararg{Int}}, dims))

# type and dimensionality specified
CLArray{T, N}(::UndefInitializer, dims::NTuple{N, Integer}) where {T, N} =
    CLArray{T, N}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T, N}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N} =
    CLArray{T, N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
CLArray{T}(::UndefInitializer, dims::NTuple{N, Integer}) where {T, N} =
    CLArray{T, N}(undef, convert(Tuple{Vararg{Int}}, dims))
CLArray{T}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N} =
    CLArray{T, N}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
CLArray{T, 1, M}() where {T, M} = CLArray{T, 1, M}(undef, 0)
CLArray{T, 1}() where {T} = CLArray{T, 1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (
        :CLArray => (),
        :(CLArray{T}) => (:T,),
        :(CLArray{T, N}) => (:T, :N),
        :(CLArray{T, N, M}) => (:T, :N, :M),
    )
    @eval begin
        function $ctor(f::Function, args...) where {$(tvars...)}
            xs = $ctor(args...)
            return try
                f(xs)
            finally
                unsafe_free!(xs)
            end
        end
    end
end

Base.similar(a::CLArray{T, N, M}) where {T, N, M} =
    CLArray{T, N, M}(undef, size(a))
Base.similar(a::CLArray{T, <:Any, M}, dims::Base.Dims{N}) where {T, N, M} =
    CLArray{T, N, M}(undef, dims)
Base.similar(a::CLArray{<:Any, <:Any, M}, ::Type{T}, dims::Base.Dims{N}) where {T, N, M} =
    CLArray{T, N, M}(undef, dims)

function Base.copy(a::CLArray{T, N}) where {T, N}
    b = similar(a)
    return @inbounds copyto!(b, a)
end

function Base.deepcopy_internal(x::CLArray, dict::IdDict)
    haskey(dict, x) && return dict[x]::typeof(x)
    return dict[x] = copy(x)
end

## array interface

Base.elsize(::Type{<:CLArray{T}}) where {T} = sizeof(T)

Base.size(x::CLArray) = x.dims
Base.sizeof(x::CLArray) = Base.elsize(x) * length(x)

context(A::CLArray) = cl.context(A.data[].mem)

memtype(x::CLArray) = memtype(typeof(x))
memtype(::Type{<:CLArray{<:Any, <:Any, M}}) where {M} = @isdefined(M) ? M : Any

is_device(a::CLArray) = memtype(a) == cl.UnifiedDeviceMemory
is_shared(a::CLArray) = memtype(a) == cl.UnifiedSharedMemory
is_host(a::CLArray) = memtype(a) == cl.UnifiedHostMemory
is_svm(a::CLArray) = memtype(a) == cl.SharedVirtualMemory
is_bda(a::CLArray) = memtype(a) == cl.BufferDeviceMemory

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
const DenseCLArray{T, N} = CLArray{T, N}
const DenseCLVector{T} = DenseCLArray{T, 1}
const DenseCLMatrix{T} = DenseCLArray{T, 2}
const DenseCLVecOrMat{T} = Union{DenseCLVector{T}, DenseCLMatrix{T}}
# XXX: these dummy aliases (DenseCLArray=CLArray) break alias printing, as
#      `Base.print_without_params` only handles the case of a single alias.

# strided arrays
const StridedSubCLArray{
    T, N, M, I <: Tuple{
        Vararg{
            Union{
                Base.RangeIndex, Base.ReshapedUnitRange,
                Base.AbstractCartesianIndex,
            },
        },
    },
} =
    SubArray{T, N, <:CLArray{<:Any,<:Any,M}, I}
const StridedCLArray{T, N, M} = Union{CLArray{T, N, M}, StridedSubCLArray{T, N, M}}
const StridedCLVector{T, M} = StridedCLArray{T, 1, M}
const StridedCLMatrix{T, M} = StridedCLArray{T, 2, M}
const StridedCLVecOrMat{T, M} = Union{StridedCLVector{T, M}, StridedCLMatrix{T, M}}

@inline device_pointer(x::StridedCLArray{T}, i::Integer = 1) where {T} =
    Base.unsafe_convert(CLPtr{T}, x) + Base._memory_offset(x, i)
@inline host_pointer(x::StridedCLArray{T}, i::Integer = 1) where {T} =
    Base.unsafe_convert(Ptr{T}, x) + Base._memory_offset(x, i)

@inline Base.pointer(x::StridedCLArray, i::Integer = 1) = device_pointer(x, i)
@inline Base.pointer(x::StridedCLArray{<:Any,<:Any,cl.UnifiedHostMemory}, i::Integer = 1) =
    host_pointer(x, i)

# anything that's (secretly) backed by a CLArray
const WrappedCLArray{T, N} = Union{CLArray{T, N}, WrappedArray{T, N, CLArray, CLArray{T, N}}}
const WrappedCLVector{T} = WrappedCLArray{T, 1}
const WrappedCLMatrix{T} = WrappedCLArray{T, 2}
const WrappedCLVecOrMat{T} = Union{WrappedCLVector{T}, WrappedCLMatrix{T}}


## interop with other arrays

@inline function CLArray{T, N, B}(xs::AbstractArray{<:Any, N}) where {T, N, B}
    A = CLArray{T, N, B}(undef, size(xs))
    copyto!(A, convert(Array{T}, xs))
    return A
end

@inline CLArray{T, N}(xs::AbstractArray{<:Any, N}) where {T, N} =
    CLArray{T, N, memory_type()}(xs)

@inline CLArray{T, N}(xs::CLArray{<:Any, N, B}) where {T, N, B} =
    CLArray{T, N, B}(xs)

# underspecified constructors
CLArray{T}(xs::AbstractArray{S, N}) where {T, N, S} = CLArray{T, N}(xs)
(::Type{CLArray{T, N} where {T}})(x::AbstractArray{S, N}) where {S, N} = CLArray{S, N}(x)
CLArray(A::AbstractArray{T, N}) where {T, N} = CLArray{T, N}(A)

# idempotency
CLArray{T, N, B}(xs::CLArray{T, N, B}) where {T, N, B} = xs
CLArray{T, N}(xs::CLArray{T, N, B}) where {T, N, B} = xs

# Level CLro references
cl.CLRef(x::Any) = cl.CLRefArray(CLArray([x]))
cl.CLRef{T}(x) where {T} = cl.CLRefArray{T}(CLArray(T[x]))
cl.CLRef{T}() where {T} = cl.CLRefArray(CLArray{T}(undef, 1))


## conversions

Base.convert(::Type{T}, x::T) where {T <: CLArray} = x


## indexing

function Base.getindex(x::CLArray{<:Any, <:Any, <:Union{cl.UnifiedHostMemory, cl.UnifiedSharedMemory}}, I::Int)
    @boundscheck checkbounds(x, I)
    return GC.@preserve x unsafe_load(host_pointer(x, I))
end

function Base.setindex!(x::CLArray{<:Any, <:Any, <:Union{cl.UnifiedHostMemory, cl.UnifiedSharedMemory}}, v, I::Int)
    @boundscheck checkbounds(x, I)
    return GC.@preserve x unsafe_store!(host_pointer(x, I), v)
end


## interop with libraries

function Base.unsafe_convert(::Type{Ptr{T}}, x::CLArray{T}) where {T}
    if is_device(x)
        throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
    end
    return convert(Ptr{T}, x.data[]) + x.offset * Base.elsize(x)
end

function Base.unsafe_convert(::Type{CLPtr{T}}, x::CLArray{T}) where {T}
    return convert(CLPtr{T}, x.data[]) + x.offset * Base.elsize(x)
end

# when passing to OpenCL kernels with `clcall`, don't convert directly to a pointer,
# but keep the underlying memory around so that we can configure the kernel correctly.
function cl.clconvert(::Type{<:Union{Ptr, CLPtr}}, x::CLArray)
    return x.data[].mem
end


## interop with GPU arrays

function Base.unsafe_convert(::Type{CLDeviceArray{T, N, AS.CrossWorkgroup}},
                             a::CLArray{T, N}) where {T, N}
    return CLDeviceArray{T, N, AS.CrossWorkgroup}(
        size(a), reinterpret(LLVMPtr{T, AS.CrossWorkgroup}, pointer(a)),
        a.maxsize - a.offset * Base.elsize(a)
    )
end


## memory copying

synchronize(x::CLArray) = synchronize(x.data[])

typetagdata(a::Array, i = 1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
typetagdata(a::CLArray, i = 1) =
    convert(CLPtr{UInt8}, a.data[]) + a.maxsize + a.offset + i - 1

function Base.copyto!(
        dest::CLArray{T}, doffs::Integer, src::Array{T}, soffs::Integer,
        n::Integer
    ) where {T}
    (n == 0 || sizeof(T) == 0) && return dest
    @boundscheck checkbounds(dest, doffs)
    @boundscheck checkbounds(dest, doffs + n - 1)
    @boundscheck checkbounds(src, soffs)
    @boundscheck checkbounds(src, soffs + n - 1)
    unsafe_copyto!(dest, doffs, src, soffs, n)
    return dest
end

Base.copyto!(dest::DenseCLArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(
        dest::Array{T}, doffs::Integer, src::DenseCLArray{T}, soffs::Integer,
        n::Integer
    ) where {T}
    (n == 0 || sizeof(T) == 0) && return dest
    @boundscheck checkbounds(dest, doffs)
    @boundscheck checkbounds(dest, doffs + n - 1)
    @boundscheck checkbounds(src, soffs)
    @boundscheck checkbounds(src, soffs + n - 1)
    unsafe_copyto!(dest, doffs, src, soffs, n)
    return dest
end

Base.copyto!(dest::Array{T}, src::DenseCLArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(
        dest::DenseCLArray{T}, doffs::Integer, src::DenseCLArray{T}, soffs::Integer,
        n::Integer
    ) where {T}
    (n == 0 || sizeof(T) == 0) && return dest
    @boundscheck checkbounds(dest, doffs)
    @boundscheck checkbounds(dest, doffs + n - 1)
    @boundscheck checkbounds(src, soffs)
    @boundscheck checkbounds(src, soffs + n - 1)
    @assert context(dest) == context(src)
    unsafe_copyto!(dest, doffs, src, soffs, n)
    return dest
end

Base.copyto!(dest::DenseCLArray{T}, src::DenseCLArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

for (srcty, dstty) in [(:Array, :CLArray), (:CLArray, :Array), (:CLArray, :CLArray)]
    @eval begin
        function Base.unsafe_copyto!(
                dst::$dstty{T}, dst_off::Int,
                src::$srcty{T}, src_off::Int,
                N::Int; blocking::Bool = true
            ) where {T}
            nbytes = N * sizeof(T)
            nbytes == 0 && return

            device_array = $dstty == CLArray ? dst : src
            cl.context!(context(device_array)) do
                if memtype(device_array) == cl.SharedVirtualMemory
                    cl.enqueue_svm_copy(pointer(dst, dst_off), pointer(src, src_off), nbytes; blocking)
                elseif memtype(device_array) == cl.BufferDeviceMemory
                    dstptr = dst isa Array ? pointer(dst) : dst.data[].mem.id
                    srcptr = src isa Array ? pointer(src) : src.data[].mem.id
                    cl.enqueue_bda_copy(dstptr, (dst_off - 1) * sizeof(T), srcptr, (src_off - 1) * sizeof(T), nbytes; blocking)
                else
                    cl.enqueue_usm_copy(pointer(dst, dst_off), pointer(src, src_off), nbytes; blocking)
                end
            end
        end
        Base.unsafe_copyto!(dst::$dstty, src::$srcty, N; kwargs...) =
            unsafe_copyto!(dst, 1, src, 1, N; kwargs...)
    end
end


## gpu array adaptor

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CLArray}, xs::AT) where {AT <: AbstractArray} =
    isbitstype(AT) ? xs : convert(CLArray, xs)

# if specific type parameters are specified, preserve those
Adapt.adapt_storage(::Type{<:CLArray{T}}, xs::AT) where {T, AT <: AbstractArray} =
    isbitstype(AT) ? xs : convert(CLArray{T}, xs)
Adapt.adapt_storage(::Type{<:CLArray{T, N}}, xs::AT) where {T, N, AT <: AbstractArray} =
    isbitstype(AT) ? xs : convert(CLArray{T, N}, xs)
Adapt.adapt_storage(::Type{<:CLArray{T, N, M}}, xs::AT) where {T, N, M, AT <: AbstractArray} =
    isbitstype(AT) ? xs : convert(CLArray{T, N, M}, xs)


## utilities

zeros(T::Type, dims...) = fill!(CLArray{T}(undef, dims...), zero(T))
ones(T::Type, dims...) = fill!(CLArray{T}(undef, dims...), one(T))
zeros(dims...) = zeros(Float32, dims...)
ones(dims...) = ones(Float32, dims...)
fill(v, dims...) = fill!(CLArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(CLArray{typeof(v)}(undef, dims...), v)

function Base.fill!(A::DenseCLArray{T}, val) where {T}
    cl.context!(context(A)) do
        GC.@preserve A begin
            if memtype(A) == cl.SharedVirtualMemory
                cl.enqueue_svm_fill(pointer(A), convert(T, val), length(A))
            elseif memtype(A) == cl.BufferDeviceMemory
                cl.enqueue_fill(A.data[].mem.id, convert(T, val), length(A))
            else
                cl.enqueue_usm_fill(pointer(A), convert(T, val), length(A))
            end
        end
    end
    return A
end


## views

context(a::SubArray) = context(parent(a))

# pointer conversions
function Base.unsafe_convert(::Type{CLPtr{T}}, V::SubArray{T, N, P, <:Tuple{Vararg{Base.RangeIndex}}}) where {T, N, P}
    return Base.unsafe_convert(CLPtr{T}, parent(V)) +
        Base._memory_offset(V.parent, map(first, V.indices)...)
end
function Base.unsafe_convert(::Type{CLPtr{T}}, V::SubArray{T, N, P, <:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange}}}}) where {T, N, P}
    return Base.unsafe_convert(CLPtr{T}, parent(V)) +
        (Base.first_index(V) - 1) * sizeof(T)
end


## PermutedDimsArray

context(a::Base.PermutedDimsArray) = context(parent(a))

Base.unsafe_convert(::Type{CLPtr{T}}, A::PermutedDimsArray) where {T} =
    Base.unsafe_convert(CLPtr{T}, parent(A))


## unsafe_wrap

"""
    unsafe_wrap(Array, arr::CLArray{_,_,cl.UnifiedSharedMemory})

Wrap a Julia `Array` around the buffer that backs a `CLArray`. This is only possible if the
GPU array is backed by a shared buffer, i.e. if it was created with `CLArray{T}(undef, ...)`.
"""
function Base.unsafe_wrap(::Type{Array}, arr::CLArray{T, N, cl.UnifiedSharedMemory}) where {T, N}
    # TODO: can we make this more convenient by increasing the buffer's refcount and using
    #       a finalizer on the Array? does that work when taking views etc of the Array?
    ptr = reinterpret(Ptr{T}, pointer(arr))
    return unsafe_wrap(Array, ptr, size(arr))
end


## resizing

"""
  resize!(a::CLVector, n::Integer)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.
"""
function Base.resize!(a::CLVector{T}, n::Integer) where {T}
    n == length(a) && return a

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
    new_data = cl.context!(context(a)) do
        mem = alloc(memtype(a), bufsize; alignment=Base.datatype_alignment(T))
        ptr = convert(CLPtr{T}, mem)
        m = min(length(a), n)
        if m > 0
            GC.@preserve a begin
                if memtype(a) == cl.SharedVirtualMemory
                    cl.enqueue_svm_copy(ptr, pointer(a), m*sizeof(T); blocking=false)
                elseif memtype(a) == cl.BufferDeviceMemory
                    cl.enqueue_bda_copy(mem.mem.id, a.data[].mem.id, m*sizeof(T); blocking=false)
                else
                    cl.enqueue_usm_copy(ptr, pointer(a), m*sizeof(T); blocking=false)
                end
            end
        end
        DataRef(free, mem)
    end
    unsafe_free!(a)

    a.data = new_data
    a.dims = (n,)
    a.maxsize = maxsize
    a.offset = 0

    return a
end
