# pointer types

export CLPtr, CL_NULL, PtrOrCLPtr, CLRef, RefOrCLRef


#
# Device pointer
#

"""
    CLPtr{T}

A memory address that refers to data of type `T` that is accessible from q device. A `CLPtr`
is ABI compatible with regular `Ptr` objects, e.g. it can be used to `ccall` a function that
expects a `Ptr` to device memory, but it prevents erroneous conversions between the two.
"""
CLPtr

if sizeof(Ptr{Cvoid}) == 8
    primitive type CLPtr{T} 64 end
else
    primitive type CLPtr{T} 32 end
end

# constructor
CLPtr{T}(x::Union{Int, UInt, CLPtr}) where {T} = Base.bitcast(CLPtr{T}, x)

const CL_NULL = CLPtr{Cvoid}(0)


## getters

Base.eltype(::Type{<:CLPtr{T}}) where {T} = T


## conversions

# to and from integers
## pointer to integer
Base.convert(::Type{T}, x::CLPtr) where {T <: Integer} = T(UInt(x))
## integer to pointer
Base.convert(::Type{CLPtr{T}}, x::Union{Int, UInt}) where {T} = CLPtr{T}(x)
Int(x::CLPtr) = Base.bitcast(Int, x)
UInt(x::CLPtr) = Base.bitcast(UInt, x)

# between regular and OpenCL pointers
Base.convert(::Type{<:Ptr}, p::CLPtr) =
    throw(ArgumentError("cannot convert a device pointer to a host pointer"))

# between OpenCL pointers
Base.convert(::Type{CLPtr{T}}, p::CLPtr) where {T} = Base.bitcast(CLPtr{T}, p)

# defer conversions to unsafe_convert
Base.cconvert(::Type{<:CLPtr}, x) = x

# fallback for unsafe_convert
Base.unsafe_convert(::Type{P}, x::CLPtr) where {P <: CLPtr} = convert(P, x)

# from arrays
Base.unsafe_convert(::Type{CLPtr{S}}, a::AbstractArray{T}) where {S, T} =
    convert(CLPtr{S}, Base.unsafe_convert(CLPtr{T}, a))
Base.unsafe_convert(::Type{CLPtr{T}}, a::AbstractArray{T}) where {T} =
    error("conversion to pointer not defined for $(typeof(a))")

## limited pointer arithmetic & comparison

Base.isequal(x::CLPtr, y::CLPtr) = (x === y)
Base.isless(x::CLPtr{T}, y::CLPtr{T}) where {T} = x < y

Base.:(==)(x::CLPtr, y::CLPtr) = UInt(x) == UInt(y)
Base.:(<)(x::CLPtr, y::CLPtr) = UInt(x) < UInt(y)
Base.:(-)(x::CLPtr, y::CLPtr) = UInt(x) - UInt(y)

Base.:(+)(x::CLPtr, y::Integer) = oftype(x, Base.add_ptr(UInt(x), (y % UInt) % UInt))
Base.:(-)(x::CLPtr, y::Integer) = oftype(x, Base.sub_ptr(UInt(x), (y % UInt) % UInt))
Base.:(+)(x::Integer, y::CLPtr) = y + x


#
# Host or device pointer
#

"""
    PtrOrCLPtr{T}

A special pointer type, ABI-compatible with both `Ptr` and `CLPtr`, for use in `ccall`
expressions to convert values to either a device or a host type (in that order). This is
required for APIs which accept pointers that either point to host or device memory.
"""
PtrOrCLPtr


if sizeof(Ptr{Cvoid}) == 8
    primitive type PtrOrCLPtr{T} 64 end
else
    primitive type PtrOrCLPtr{T} 32 end
end

function Base.cconvert(::Type{PtrOrCLPtr{T}}, val) where {T}
    # `cconvert` is always implemented for both `Ptr` and `CLPtr`, so pick the first result
    # that has done an actual conversion

    dev_val = Base.cconvert(CLPtr{T}, val)
    if dev_val !== val
        return dev_val
    end

    host_val = Base.cconvert(Ptr{T}, val)
    if host_val !== val
        return host_val
    end

    return val
end

function Base.unsafe_convert(::Type{PtrOrCLPtr{T}}, val) where {T}
    ptr = if Core.Compiler.return_type(
            Base.unsafe_convert,
            Tuple{Type{Ptr{T}}, typeof(val)}
        ) !== Union{}
        Base.unsafe_convert(Ptr{T}, val)
    elseif Core.Compiler.return_type(
            Base.unsafe_convert,
            Tuple{Type{CLPtr{T}}, typeof(val)}
        ) !== Union{}
        Base.unsafe_convert(CLPtr{T}, val)
    else
        throw(ArgumentError("cannot convert to either a host or device pointer"))
    end

    return Base.bitcast(PtrOrCLPtr{T}, ptr)
end


#
# Device reference objects
#

if sizeof(Ptr{Cvoid}) == 8
    primitive type CLRef{T} 64 end
else
    primitive type CLRef{T} 32 end
end

# general methods for CLRef{T} type
Base.eltype(x::Type{<:CLRef{T}}) where {T} = @isdefined(T) ? T : Any

Base.convert(::Type{CLRef{T}}, x::CLRef{T}) where {T} = x

# conversion or the actual ccall
Base.unsafe_convert(::Type{CLRef{T}}, x::CLRef{T}) where {T} = Base.bitcast(CLRef{T}, Base.unsafe_convert(CLPtr{T}, x))
Base.unsafe_convert(::Type{CLRef{T}}, x) where {T} = Base.bitcast(CLRef{T}, Base.unsafe_convert(CLPtr{T}, x))

# CLRef from literal pointer
Base.convert(::Type{CLRef{T}}, x::CLPtr{T}) where {T} = x

# indirect constructors using CLRef
Base.convert(::Type{CLRef{T}}, x) where {T} = CLRef{T}(x)


## CLRef object backed by an array at index i

struct CLRefArray{T, A <: AbstractArray{T}} <: Ref{T}
    x::A
    i::Int
    CLRefArray{T, A}(x, i) where {T, A <: AbstractArray{T}} = new(x, i)
end
CLRefArray{T}(x::AbstractArray{T}, i::Int = 1) where {T} = CLRefArray{T, typeof(x)}(x, i)
CLRefArray(x::AbstractArray{T}, i::Int = 1) where {T} = CLRefArray{T}(x, i)
Base.convert(::Type{CLRef{T}}, x::AbstractArray{T}) where {T} = CLRefArray(x, 1)

function Base.unsafe_convert(P::Type{CLPtr{T}}, b::CLRefArray{T}) where {T}
    return pointer(b.x, b.i)
end
function Base.unsafe_convert(P::Type{CLPtr{Any}}, b::CLRefArray{Any})
    return convert(P, pointer(b.x, b.i))
end
Base.unsafe_convert(::Type{CLPtr{Cvoid}}, b::CLRefArray{T}) where {T} =
    convert(CLPtr{Cvoid}, Base.unsafe_convert(CLPtr{T}, b))


## Union with all CLRef 'subtypes'

const CLRefs{T} = Union{CLPtr{T}, CLRefArray{T}}


## RefOrCLRef

if sizeof(Ptr{Cvoid}) == 8
    primitive type RefOrCLRef{T} 64 end
else
    primitive type RefOrCLRef{T} 32 end
end

Base.convert(::Type{RefOrCLRef{T}}, x::Union{RefOrCLRef{T}, Ref{T}, CLRef{T}, CLRefs{T}}) where {T} = x

# prefer conversion to CPU ref: this is generally cheaper
Base.convert(::Type{RefOrCLRef{T}}, x) where {T} = Ref{T}(x)
Base.unsafe_convert(::Type{RefOrCLRef{T}}, x::Ref{T}) where {T} =
    Base.bitcast(RefOrCLRef{T}, Base.unsafe_convert(Ptr{T}, x))
Base.unsafe_convert(::Type{RefOrCLRef{T}}, x) where {T} =
    Base.bitcast(RefOrCLRef{T}, Base.unsafe_convert(Ptr{T}, x))

# support conversion from GPU ref
Base.unsafe_convert(::Type{RefOrCLRef{T}}, x::CLRefs{T}) where {T} =
    Base.bitcast(RefOrCLRef{T}, Base.unsafe_convert(CLPtr{T}, x))

# support conversion from arrays
Base.convert(::Type{RefOrCLRef{T}}, x::Array{T}) where {T} = convert(Ref{T}, x)
Base.convert(::Type{RefOrCLRef{T}}, x::AbstractArray{T}) where {T} = convert(CLRef{T}, x)
Base.unsafe_convert(P::Type{RefOrCLRef{T}}, b::CLRefArray{T}) where {T} =
    Base.bitcast(RefOrCLRef{T}, Base.unsafe_convert(CLRef{T}, b))
