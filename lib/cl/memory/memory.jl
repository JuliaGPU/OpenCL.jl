# Raw memory management

export device_alloc, host_alloc, shared_alloc, svm_alloc, free

#
# untyped buffers
#

abstract type AbstractMemory end

Base.convert(T::Type{<:Union{Ptr, CLPtr}}, buf::AbstractMemory) =
    throw(ArgumentError("Illegal conversion of a $(typeof(buf)) to a $T"))

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(P::Type{<:Union{Ptr, CLPtr}}, buf::AbstractMemory) = convert(P, buf)

include("usm.jl")
include("svm.jl")
