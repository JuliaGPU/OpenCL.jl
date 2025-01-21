export OutOfGPUMemoryError

"""
    OutOfGPUMemoryError()

An operation allocated too much GPU memory.
"""
struct OutOfGPUMemoryError <: Exception
    sz::Int
    dev::cl.Device

    function OutOfGPUMemoryError(sz::Integer = 0, dev::cl.Device = cl.device())
        return new(sz, dev)
    end
end

function Base.showerror(io::IO, err::OutOfGPUMemoryError)
    print(io, "Out of GPU memory")
    if err.sz > 0
        print(io, " trying to allocate $(Base.format_bytes(err.sz))")
    end
    print(" on device $((err.dev).name)")
    #=
    if length(memory_properties(err.dev)) == 1
        # XXX: how to handle multiple memories?
        print(" with $(Base.format_bytes(only(memory_properties(err.dev)).totalSize))")
    end
    =#
    return io
end

function allocate(::Type{cl.DeviceBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return cl.DeviceBuffer(cl.CL_NULL, bytes, ctx, dev)

    buf = cl.device_alloc(ctx, dev, bytes, alignment = alignment)
    # make_resident(ctx, dev, buf)
    return buf
end

function allocate(::Type{cl.SharedBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return cl.SharedBuffer(cl.CL_NULL, bytes, ctx, dev)

    # TODO: support cross-device shared buffers (by setting `dev=nothing`)

    buf = cl.shared_alloc(ctx, dev, bytes, alignment = alignment)

    return buf
end

function allocate(::Type{cl.HostBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return cl.HostBuffer(cl.CL_NULL, bytes, ctx)
    return cl.host_alloc(ctx, bytes, alignment = alignment)
end

function allocate(::Type{cl.SVMBuffer}, ctx, dev, bytes::Int, alignment::Int)
    bytes == 0 && return cl.SVMBuffer(cl.CL_NULL, bytes, ctx)

    buf = cl.svm_alloc(ctx, bytes, alignment = alignment)
    # make_resident(ctx, dev, buf)
    return buf
end

function release(buf::cl.AbstractBuffer)
    sizeof(buf) == 0 && return

    # XXX: is it necessary to evice memory if we are going to free it?
    #      this is racy, because eviction is not queue-ordered, and
    #      we don't want to synchronize inside what could have been a
    #      GC-driven finalizer. if we need to, port the stream/queue
    #      tracking from CUDA.jl so that we can synchronize only the
    #      queue that's associated with the buffer.
    #if buf isa oneL0.DeviceBuffer || buf isa oneL0.SharedBuffer
    #    ctx = oneL0.context(buf)
    #    dev = oneL0.device(buf)
    #    evict(ctx, dev, buf)
    #end

    free(buf, blocking = true)

    # TODO: queue-ordered free from non-finalizer tasks once we have
    #       `zeMemFreeAsync(ptr, queue)`

    return
end
