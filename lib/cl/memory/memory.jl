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
include("backend.jl")

############################################

# free function for different buffers

function free(buf::AbstractMemory; blocking = false)
    ctx = context(buf)
    ptr = Ptr{Nothing}(UInt(buf.ptr))
    if buf isa SharedVirtualMemory
        clSVMFree(ctx, ptr)
        if blocking
            finish(queue())
        end
    else
        if blocking
            clMemBlockingFreeINTEL(ctx, ptr)
        else
            clMemFreeINTEL(ctx, ptr)
        end
    end
    return
end

#############################################

# generic memory operations for different buffers

function enqueue_abstract_memcpy(
        dst::Union{Ptr, CLPtr}, src::Union{Ptr, CLPtr}, nbytes::Integer; queu::CmdQueue = queue(), blocking::Bool = false,
        wait_for::Vector{Event} = Event[], backend = select_backend()
    )
    return if backend == USMBackend
        enqueue_usm_memcpy(dst, src, nbytes; queu = queu, blocking = blocking, wait_for = wait_for)
    elseif backend == SVMBackend
        enqueue_svm_memcpy(dst, src, nbytes; queu = queu, blocking = blocking, wait_for = wait_for)
    end
end

function enqueue_abstract_fill(ptr::Union{Ptr, CLPtr}, pattern::Union{Ptr, CLPtr}, pattern_size::Integer, nbytes::Integer; queu::CmdQueue = queue(), wait_for::Vector{Event} = Event[], backend = select_backend())
    return if backend == USMBackend
        enqueue_usm_memfill(ptr, pattern, pattern_size, nbytes; queu = queu, wait_for = wait_for)
    elseif backend == SVMBackend
        enqueue_svm_fill(ptr, pattern, pattern_size, nbytes; queu = queu, wait_for = wait_for)
    end
end
