export CLArray, CLVector, CLMatrix, CLVecOrMat,
       device_accessible, host_accessible


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
    return !("cl_khr_fp64" in cl.device().extensions) && contains_eltype(T, Float64) && error("Float64 is not supported on this device")
end

mutable struct CLArray{T, N, M} <: AbstractGPUArray{T, N}
    data::DataRef{Managed{M}}

    maxsize::Int  # maximum data size; excluding any selector bytes
    offset::Int   # offset of the data in memory, in bytes

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
            buf = managed_alloc(M, bufsize; alignment=Base.datatype_alignment(T))
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
    elseif cl.memory_backend() == cl.BufferBackend()
        return cl.Buffer
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

# can we read this array from the device (i.e. derive a CLPtr)?
device_accessible(a::CLArray) =
    memtype(a) in (cl.UnifiedDeviceMemory, cl.UnifiedSharedMemory, cl.SharedVirtualMemory, cl.Buffer)
host_accessible(a::CLArray) =
    memtype(a) in (cl.UnifiedHostMemory, cl.UnifiedSharedMemory, cl.SharedVirtualMemory)


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

function Base.getindex(x::CLArray{<:Any, <:Any, <:Union{cl.UnifiedHostMemory, cl.UnifiedSharedMemory, cl.SharedVirtualMemory}}, I::Int)
    @boundscheck checkbounds(x, I)
    return Base.@lock x.data[].lock begin
        GC.@preserve x unsafe_load(host_pointer(x, I))
    end
end

function Base.setindex!(x::CLArray{<:Any, <:Any, <:Union{cl.UnifiedHostMemory, cl.UnifiedSharedMemory, cl.SharedVirtualMemory}}, v, I::Int)
    @boundscheck checkbounds(x, I)
    return Base.@lock x.data[].lock begin
        GC.@preserve x unsafe_store!(host_pointer(x, I), v)
    end
end


## interop with libraries

function Base.unsafe_convert(::Type{Ptr{T}}, x::CLArray{T}) where {T}
    if !host_accessible(x)
        throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
    end
    return convert(Ptr{T}, x.data[]) + x.offset
end

function Base.unsafe_convert(::Type{CLPtr{T}}, x::CLArray{T}) where {T}
    if !device_accessible(x)
        throw(ArgumentError("cannot take the device address of a $(typeof(x))"))
    end
    return convert(CLPtr{T}, x.data[]) + x.offset
end

# when passing to OpenCL kernels with `clcall`, don't convert directly to a pointer, but
# to the array itself: `convert_arguments` keeps the converted values alive and locked
# through enqueue, so returning the owner both preserves the `DataRef` (preventing
# finalization of the allocation mid-launch) and carries the allocation lock.
# the pointer is only extracted by `unsafe_clconvert`, under that lock.
cl.clconvert(::Type{<:Union{Ptr, CLPtr}}, x::CLArray) = x
cl.argument_lock(x::CLArray) = x.data[].lock
function cl.unsafe_clconvert(typ::Type{<:Union{Ptr, CLPtr}}, x::CLArray{<:Any, <:Any, <:cl.AbstractMemoryObject})
    device_convert(cl.Buffer, x.data[])
    return x.data[].mem
end
function cl.unsafe_clconvert(typ::Type{<:Union{Ptr{T}, CLPtr{T}}}, x::CLArray{<:Any, <:Any, M}) where {T,M<:cl.AbstractPointerMemory}
    ptr = device_convert(typ, x.data[])
    return cl.TrackedPtr{T,M}(ptr)
end


## interop with GPU arrays

function Base.unsafe_convert(::Type{CLDeviceArray{T, N, AS.CrossWorkgroup}},
                             a::CLArray{T, N}) where {T, N}
    return CLDeviceArray{T, N, AS.CrossWorkgroup}(
        size(a), reinterpret(LLVMPtr{T, AS.CrossWorkgroup}, pointer(a)),
        a.maxsize - a.offset
    )
end


## memory copying

synchronize(x::CLArray) = synchronize(x.data[])

typetagdata(a::Array, i = 1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
function typetagdata(a::CLArray, i = 1)
    # for zero-size element types (e.g. singleton unions), the byte offset
    # is always zero, so the corresponding element offset is also zero
    elem_offset = iszero(Base.elsize(a)) ? 0 : a.offset ÷ Base.elsize(a)
    return convert(CLPtr{UInt8}, a.data[]) + a.maxsize + elem_offset + i - 1
end

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

# the host address of an array operand in a copy. for arrays wrapping host memory, the
# device pointer is used, which is identical but also records the device-side use.
host_address(a::Array, i) = pointer(a, i)
host_address(a::CLArray{T}, i) where {T} = reinterpret(Ptr{T}, pointer(a, i))

for (srcty, dstty) in [(:Array, :CLArray), (:CLArray, :Array), (:CLArray, :CLArray)]
    @eval begin
        function Base.unsafe_copyto!(
                dst::$dstty{T}, dst_off::Int,
                src::$srcty{T}, src_off::Int,
                N::Int; blocking::Bool = true
            ) where {T}
            nbytes = N * sizeof(T)
            nbytes == 0 && return

            # arrays wrapping host memory can be treated like host arrays, so dispatch on
            # the memory type of the other array
            device_array = if $dstty == CLArray && $srcty == CLArray
                is_system(dst) ? src : dst
            else
                $dstty == CLArray ? dst : src
            end
            cl.context!(context(device_array)) do
                managed = Managed[]
                dst isa CLArray && push!(managed, dst.data[])
                src isa CLArray && push!(managed, src.data[])
                # the copy has to be submitted before the arrays can be freed, which for
                # arrays wrapping host memory also releases that memory
                GC.@preserve dst src with_managed_locks(managed) do
                    if memtype(device_array) == cl.SharedVirtualMemory
                        cl.enqueue_svm_copy(pointer(dst, dst_off), pointer(src, src_off), nbytes; blocking)
                    elseif memtype(device_array) <: cl.UnifiedMemory
                        cl.enqueue_usm_copy(pointer(dst, dst_off), pointer(src, src_off), nbytes; blocking)
                    else
                        dst_buffer = dst isa CLArray && !is_system(dst)
                        src_buffer = src isa CLArray && !is_system(src)
                        if dst_buffer && src_buffer
                            cl.enqueue_copy(convert(cl.Buffer, dst.data[]),
                                dst.offset + (dst_off - 1) * sizeof(T),
                                convert(cl.Buffer, src.data[]),
                                src.offset + (src_off - 1) * sizeof(T),
                                nbytes; blocking)
                        elseif dst_buffer
                            cl.enqueue_write(convert(cl.Buffer, dst.data[]),
                                dst.offset + (dst_off - 1) * sizeof(T),
                                host_address(src, src_off), nbytes; blocking)
                        elseif src_buffer
                            cl.enqueue_read(host_address(dst, dst_off),
                                convert(cl.Buffer, src.data[]),
                                src.offset + (src_off - 1) * sizeof(T),
                                nbytes; blocking)
                        end
                    end
                end
                # a blocking copy completes the kernels queued before it, so surface any
                # exception they threw rather than returning their garbage
                blocking && cl.finish(cl.queue())
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

fill_aligned(A::CLArray{T, <:Any, <:cl.AbstractPointerMemory}) where {T} =
    iszero((UInt(convert(CLPtr{T}, A.data[].mem)) + A.offset) % sizeof(T))
fill_aligned(A::CLArray{T}) where {T} = iszero(A.offset % sizeof(T))

function Base.fill!(A::DenseCLArray{T}, val) where {T}
    isempty(A) && return A
    # the OpenCL fill commands only accept patterns of 1, 2, 4, ..., 128 bytes, and
    # require the destination to be aligned to that size, so fall back to a kernel
    # otherwise. we also do so for host memory wrapped using `unsafe_wrap`, which
    # implementations do not expect here (PoCL crashes).
    if !ispow2(sizeof(T)) || sizeof(T) > 128 || !fill_aligned(A) || is_system(A)
        return invoke(fill!, Tuple{AnyGPUArray, Any}, A, val)
    end
    cl.context!(context(A)) do
        Base.@lock A.data[].lock begin
            GC.@preserve A begin
                if memtype(A) == cl.SharedVirtualMemory
                    cl.enqueue_svm_fill(pointer(A), convert(T, val), length(A))
                elseif memtype(A) <: cl.UnifiedMemory
                    cl.enqueue_usm_fill(pointer(A), convert(T, val), length(A))
                else
                    cl.enqueue_fill(convert(cl.Buffer, A.data[]), A.offset, convert(T, val), length(A))
                end
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
    unsafe_wrap(Array, arr::CLArray)

Wrap a Julia `Array` around the memory that backs a `CLArray`, without copying. This is
only possible if that memory is accessible from the host: unified host or shared memory,
shared virtual memory, or host memory that was itself wrapped using
`unsafe_wrap(CLArray, ...)`.

Device operations execute asynchronously, so wait for them to finish (e.g., using
`cl.finish(cl.queue())`) before accessing the returned array after using `arr` on the
device.

!!! warning

    The returned `Array` does **not** keep `arr` alive. If `arr` is garbage collected (or
    freed using `unsafe_free!`), its memory is released and the `Array` refers to invalid
    memory. The caller must keep a reference to `arr` for as long as the `Array`, or
    anything derived from it, is used.

!!! warning

    Coarse-grained shared virtual memory is only accessible from the host while it is
    mapped. Using `arr` on the device unmaps it, after which the returned array must not
    be accessed until an operation on `arr` has mapped it again (e.g., indexing it, or
    calling `unsafe_wrap(Array, arr)` again).
"""
function Base.unsafe_wrap(::Type{Array}, arr::CLArray{T, N}) where {T, N}
    return unsafe_wrap(Array, host_pointer(arr), size(arr))
end

"""
    unsafe_wrap(CLArray, a::Array)
    unsafe_wrap(CLArray, ptr::Ptr{T}, dims)
    unsafe_wrap(CLArray{T,N,M}, ...)

Wrap a `CLArray` around host memory, without copying, so that it can be used on the
device, e.g., in kernels or broadcasts. Changes made through the `CLArray` are visible in
the original array, and vice versa.

This requires a device that can directly access ordinary host memory, i.e., one that
supports fine-grained system shared virtual memory (like PoCL's CPU device), or shared
system allocations in the `cl_intel_unified_shared_memory` extension. On other devices an
`ArgumentError` is thrown; use `CLArray(a)` to copy the data instead. The memory type `M`
of the resulting array is either `cl.SharedVirtualMemory` or `cl.UnifiedSharedMemory`,
depending on what the device supports. It can be selected explicitly by passing the full
array type.

When wrapping an `Array`, the returned `CLArray` keeps it alive. When wrapping a pointer,
the caller has to make sure the memory stays valid for as long as the `CLArray` is used.
In both cases, the memory must not be freed or reallocated while it is wrapped (e.g., by
calling `resize!` on the original array), and resizing the wrapper is not supported.

8- and 16-bit atomic operations access the entire aligned 4-byte word that contains the
element, so when using them, every such word that overlaps the wrapped memory must be
accessible, even if it extends past the end of that memory. `CLArray`s allocated by OpenCL.jl
are padded to guarantee this.

Device operations execute asynchronously, so wait for them to finish (e.g., using
`cl.finish(cl.queue())`) before accessing the original memory on the host. Wrapping the
same memory multiple times results in independent arrays whose operations are not
synchronized with each other.

```julia
a = rand(Float32, 1024)
b = unsafe_wrap(CLArray, a)
b .= sin.(b)        # executes on the device, updating `a`
cl.finish(cl.queue())
```
"""
unsafe_wrap(::Type{<:CLArray}, ::Any, ::Any...)

function system_usm_supported(dev::cl.Device)
    cl.usm_supported(dev) || return false
    caps = cl.usm_capabilities(dev)
    # the resulting arrays are typed as shared memory, so operations like `similar` also
    # need to be able to allocate regular shared memory.
    return caps.shared.access && caps.single_device.access
end

function system_svm_supported(dev::cl.Device)
    caps = cl.svm_capabilities(dev)
    # operations like `similar` need to be able to allocate regular SVM
    return caps.fine_grain_system && caps.coarse_grain_buffer
end

# the memory type to use for wrapping system memory, or `nothing` if not supported
function system_memory_type(dev::cl.Device = cl.device())
    usm = system_usm_supported(dev)
    svm = system_svm_supported(dev)
    # stick to the same extension as regular allocations, if possible
    if usm && (!svm || cl.memory_backend() == cl.USMBackend())
        return cl.UnifiedSharedMemory
    elseif svm
        return cl.SharedVirtualMemory
    else
        return nothing
    end
end

# `owner` is kept alive for as long as the wrapper
function wrap_system_memory(::Type{CLArray{T, N, M}}, ptr::Ptr{T}, dims::NTuple{N, Int},
                            owner = nothing) where {T, N, M}
    check_eltype(T)
    isbitstype(T) || throw(ArgumentError("Can only unsafe_wrap a pointer to a bits type"))
    all(>=(0), dims) || throw(ArgumentError("Invalid dimensions $dims"))
    bytesize = Base.checked_mul(foldl(Base.checked_mul, dims; init = 1), sizeof(T))
    if bytesize > 0 && ptr == C_NULL
        throw(ArgumentError("Cannot wrap a NULL pointer"))
    end
    if !iszero(UInt(ptr) % Base.datatype_alignment(T))
        throw(ArgumentError("Pointer $ptr is not sufficiently aligned for elements of type $T"))
    end

    dev = cl.device()
    supported = if M == cl.SharedVirtualMemory
        system_svm_supported(dev)
    elseif M == cl.UnifiedSharedMemory
        system_usm_supported(dev)
    else
        throw(ArgumentError("Cannot wrap host memory as $M; use cl.SharedVirtualMemory or cl.UnifiedSharedMemory"))
    end
    supported || throw(ArgumentError("Device $(dev.name) does not support accessing host memory as $M"))

    mem = M(reinterpret(CLPtr{Cvoid}, ptr), bytesize, cl.context(), true)
    # the memory is only ever freed by its owner, but we still need to synchronize
    # outstanding operations before releasing it (and our reference to the queue)
    data = DataRef(Managed(mem; dirty = false)) do managed
        GC.@preserve owner free(managed)
    end
    return CLArray{T, N}(data, dims)
end
function wrap_system_memory(::Type{CLArray{T, N}}, ptr::Ptr{T}, dims::NTuple{N, Int},
                            owner = nothing) where {T, N}
    M = system_memory_type()
    if M === nothing
        throw(ArgumentError("""Device $(cl.device().name) cannot access host memory directly, which is required to wrap it as a CLArray.
                               Use `CLArray(a)` to copy the data instead."""))
    end
    return wrap_system_memory(CLArray{T, N, M}, ptr, dims, owner)
end

Base.unsafe_wrap(::Union{Type{CLArray}, Type{CLArray{T}}, Type{CLArray{T, N}}},
                 ptr::Ptr{T}, dims::NTuple{N, Int}) where {T, N} =
    wrap_system_memory(CLArray{T, N}, ptr, dims)
Base.unsafe_wrap(::Type{CLArray{T, N, M}}, ptr::Ptr{T}, dims::NTuple{N, Int}) where {T, N, M} =
    wrap_system_memory(CLArray{T, N, M}, ptr, dims)

# integer size input
Base.unsafe_wrap(::Union{Type{CLArray}, Type{CLArray{T}}, Type{CLArray{T, 1}}},
                 ptr::Ptr{T}, dim::Integer) where {T} =
    unsafe_wrap(CLArray{T, 1}, ptr, (Int(dim),))
Base.unsafe_wrap(::Type{CLArray{T, 1, M}}, ptr::Ptr{T}, dim::Integer) where {T, M} =
    unsafe_wrap(CLArray{T, 1, M}, ptr, (Int(dim),))

# array input: keep the array alive for as long as the wrapper
Base.unsafe_wrap(::Union{Type{CLArray}, Type{CLArray{T}}, Type{CLArray{T, N}}},
                 a::Array{T, N}) where {T, N} =
    wrap_system_memory(CLArray{T, N}, pointer(a), size(a), a)
Base.unsafe_wrap(::Type{CLArray{T, N, M}}, a::Array{T, N}) where {T, N, M} =
    wrap_system_memory(CLArray{T, N, M}, pointer(a), size(a), a)

# whether an array wraps host memory using `unsafe_wrap`
is_system(a::CLArray) = cl.is_system(a.data[].mem)


## resizing

"""
  resize!(a::CLVector, n::Integer)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.
"""
function Base.resize!(a::CLVector{T}, n::Integer) where {T}
    n == length(a) && return a
    # resizing would detach the array from the host memory it wraps
    is_system(a) && throw(ArgumentError("Cannot resize a CLArray that wraps host memory"))

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
        mem = managed_alloc(memtype(a), bufsize; alignment=Base.datatype_alignment(T))
        with_managed_locks(Managed[mem, a.data[]]) do
            ptr = convert(CLPtr{T}, mem)
            m = min(length(a), n)
            if m > 0
                GC.@preserve a begin
                    if memtype(a) == cl.SharedVirtualMemory
                        cl.enqueue_svm_copy(ptr, pointer(a), m*sizeof(T); blocking=false)
                    elseif memtype(a) <: cl.UnifiedMemory
                        cl.enqueue_usm_copy(ptr, pointer(a), m*sizeof(T); blocking=false)
                    else
                        cl.enqueue_copy(convert(cl.Buffer, mem), 0, convert(cl.Buffer, a.data[]), a.offset, m*sizeof(T); blocking=false)
                    end
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
