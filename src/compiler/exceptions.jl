# support for device-side exceptions

## exception type

"""
    KernelException

An exception thrown during kernel execution on `dev`, detected when synchronizing
(`cl.finish`, a blocking copy, `Array(x)`, ...).

How much is known about the exception depends on the debug level the kernel was compiled
with (the session's `-g` level, or `@opencl debug_level=`):

- `0`: only that an exception was thrown;
- `1`: additionally its type `name` and `reason`, for exceptions thrown by Julia's runtime
  (bounds errors, domain errors, ...);
- `2`: additionally the position of the faulting work-item (`work_item` is its local id,
  `work_group` the id of its work-group), the name of any other exception, and a device-side
  `backtrace` as `(function, file, line)` tuples.

`dev` identifies the device whose mailbox reported the exception.
Fields that were not recorded are empty strings, empty vectors, or all-zero tuples.
"""
struct KernelException <: Exception
    dev::cl.Device
    name::String
    reason::String
    work_item::NTuple{3, Int}
    work_group::NTuple{3, Int}
    backtrace::Vector{Tuple{String, String, Int}}   # (function, file, line) per frame
end

# Positions are 1-based, so zero coordinates indicate that level 2 details were not recorded.
function Base.showerror(io::IO, err::KernelException)
    name = isempty(err.name) ? "exception" : err.name
    article = first(uppercase(name)) in ('A', 'E', 'I', 'O', 'U') ? "An" : "A"
    print(io, "KernelException: $article $name was thrown")
    if err.work_item != (0, 0, 0)
        work_item = join(err.work_item, '×')
        work_group = join(err.work_group, '×')
        print(io, " by work-item $work_item in work-group $work_group")
    end
    print(io, " on device ", err.dev.name)
    isempty(err.reason) || print(io, ": ", err.reason)
    if err.work_item == (0, 0, 0)
        print(io, "\nFor more details, run Julia with `-g2`, or launch the kernel with `@opencl debug_level=2`")
    else
        print(io, "\nStacktrace:")
        for (i, (func, file, line)) in enumerate(err.backtrace)
            print(io, "\n [", i, "] ", func)
            isempty(file) || print(io, " at ", file, ":", line)
        end
    end
end

# decode a null-terminated mailbox text buffer into a `String`
function exception_string(bytes::NTuple{N, UInt8}) where {N}
    len = something(findfirst(iszero, bytes), N + 1) - 1
    return String(UInt8[bytes[i] for i in 1:len])
end


## exception mailbox

# One mailbox per (context, device), shared by its queues. Track submissions so host
# access, including mapping coarse-grained memory, waits for every possible writer.
mutable struct ExceptionMailbox
    # the host-accessible memory backing the mailbox, or `nothing` when the context's
    # devices don't support any (in which case kernels get a zero address)
    const mem::Union{Nothing, cl.AbstractMemory}
    const address::UInt64
    # whether host access requires mapping the memory (coarse-grained SVM, buffers)
    const mapped::Bool
    # serialize launches and checks for this mailbox
    const lock::ReentrantLock
    # queues that have launched kernels since the mailbox was last checked on them
    const pending::Set{cl.CmdQueue}
    # assigned under the lock; distinguishes work-items from different launches
    launch_id::UInt64
end

# Device-scoped atomics cannot protect a mailbox shared by different devices. These strong
# references retain contexts and backing allocations for the lifetime of the process.
const exception_mailboxes = Dict{Tuple{cl.Context, cl.Device}, ExceptionMailbox}()
const exception_mailboxes_lock = ReentrantLock()

# the kinds of memory a mailbox can live in, in order of preference: memory the host can
# access directly comes first, memory that needs mapping around every host access last.
const exception_mailbox_backends = (:usm, :svm_fine, :svm_coarse, :buffer)

# allocate a mailbox in memory all devices in the context can reach by pointer, trying the
# given backends in order.
function allocate_exception_mailbox(ctx::cl.Context, queue::cl.CmdQueue=cl.queue();
                                    backends=exception_mailbox_backends)
    devs = ctx.devices
    sz = sizeof(ExceptionInfo_st)
    mem, mapped = cl.context!(ctx) do
        for backend in backends
            if backend === :usm
                if all(dev -> cl.usm_supported(dev) && cl.usm_capabilities(dev).host.access, devs)
                    return cl.host_alloc(sz), false
                end
            elseif backend === :svm_fine
                if all(dev -> cl.svm_capabilities(dev).fine_grain_buffer, devs)
                    return cl.svm_alloc(sz; fine_grained=true), false
                end
            elseif backend === :svm_coarse
                if all(dev -> cl.svm_capabilities(dev).coarse_grain_buffer, devs)
                    return cl.svm_alloc(sz), true
                end
            elseif backend === :buffer
                if all(cl.bda_supported, devs)
                    return cl.Buffer(sz; host_accessible=true, device_private_address=true), true
                end
            else
                throw(ArgumentError("Unknown exception mailbox backend $backend"))
            end
        end
        return nothing, false
    end

    if mem === nothing
        @warn """Device-side exceptions cannot be reported on $(join(map(dev -> dev.name, devs), ", ")): \
                 the device does not support any host-accessible memory to put the exception mailbox in.
                 Kernels that throw will complete silently."""
        return ExceptionMailbox(nothing, UInt64(0), false, ReentrantLock(),
                                Set{cl.CmdQueue}(), UInt64(0))
    end

    mailbox = ExceptionMailbox(mem, UInt64(UInt(pointer(mem))), mapped, ReentrantLock(),
                               Set{cl.CmdQueue}(), UInt64(0))
    with_exception_mailbox(mailbox, queue) do ptr
        unsafe_store!(ptr, ExceptionInfo_st())
    end
    return mailbox
end

# run `f` with a host pointer to the mailbox, mapping its memory through `queue` as needed
function with_exception_mailbox(f, mailbox::ExceptionMailbox, queue::cl.CmdQueue)
    sz = sizeof(ExceptionInfo_st)
    mem = mailbox.mem
    if mem isa cl.Buffer
        ptr, _ = cl.enqueue_map(mem, sz, :rw; queue, blocking=true)
        try
            return f(convert(Ptr{ExceptionInfo_st}, ptr))
        finally
            cl.enqueue_unmap(mem, ptr; queue)
            # A later launch may use another queue, so complete the unmap before releasing
            # the mailbox lock.
            cl.clFinish(queue)
        end
    end
    ptr = convert(Ptr{ExceptionInfo_st}, mem)
    mailbox.mapped || return f(ptr)
    clptr = convert(CLPtr{Cvoid}, mem)
    cl.enqueue_svm_map(clptr, sz, :rw; queue, blocking=true)
    try
        return f(ptr)
    finally
        cl.enqueue_svm_unmap(clptr; queue)
        # A later launch may use another queue, so complete the unmap before releasing
        # the mailbox lock.
        cl.clFinish(queue)
    end
end

# Launch while holding the mailbox lock, so a concurrent synchronization cannot finish the
# queue between enqueue and recording it as pending (or map the mailbox during submission).
# Keep this in an ordinary function: the generated `AbstractKernel` call cannot contain a
# closure or `do` block on Julia 1.13.
function launch_with_exception_mailbox(kernel::cl.Kernel, args...;
                                       indirect_memory::Vector{cl.AbstractMemory},
                                       rng_state::Bool, kwargs...)
    ctx, dev, queue = cl.context(), cl.device(), cl.queue()
    mailbox = Base.@lock exception_mailboxes_lock begin
        get!(exception_mailboxes, (ctx, dev)) do
            allocate_exception_mailbox(ctx, queue)
        end
    end
    Base.@lock mailbox.lock begin
        mailbox.mem === nothing || push!(indirect_memory, mailbox.mem)
        mailbox.launch_id += UInt64(1)
        state = KernelState(rng_state ? Base.rand(UInt32) : UInt32(0), mailbox.address,
                            mailbox.launch_id)
        result = cl.call(kernel, state, args...; indirect_memory, rng_state, kwargs...)
        if mailbox.mem !== nothing
            push!(mailbox.pending, queue)
        end
        return result
    end
end

"""
    check_exceptions(queue::cl.CmdQueue)

Synchronize `queue` and every other queue recorded against its device's mailbox, then check
whether a kernel threw an exception and, if so, rethrow it host-side as a
[`KernelException`](@ref).

The exception mailbox is shared by queues targeting the same device in a context, so this
may wait for and surface an exception from another queue on that device.
"""
function check_exceptions(queue::cl.CmdQueue; rethrow::Bool=true)
    # Finalizers cannot yield while waiting for a contended Julia lock. Finish the queue
    # without touching mailbox bookkeeping; an ordinary check will finish it again and
    # consume the report. This also keeps finalizers independent of task-local state.
    if !rethrow
        cl.clFinish(queue)
        return
    end
    mailbox = Base.@lock exception_mailboxes_lock begin
        get(exception_mailboxes, (queue.context, queue.device), nothing)
    end
    if mailbox === nothing
        cl.clFinish(queue)
        return
    end
    Base.@lock mailbox.lock begin
        # Keep this raw: `cl.finish` delegates here in order to hold the mailbox lock across
        # both synchronization and inspection.
        cl.clFinish(queue)
        isempty(mailbox.pending) && return

        # The mailbox cannot be mapped or read while another queue may still access it.
        # Finish all queues that launched with its address; the launch-side lock prevents a
        # new enqueue from appearing between this loop and the reset below.
        for pending in mailbox.pending
            pending == queue || cl.clFinish(pending)
        end

        exc = with_exception_mailbox(mailbox, queue) do ptr
            status_ptr = convert(Ptr{Int32}, ptr)
            unsafe_load(status_ptr) == 0 && return nothing
            info = unsafe_load(ptr)
            nframes = min(Int(info.num_frames), EXCEPTION_MAX_FRAMES)
            backtrace = Tuple{String, String, Int}[
                (exception_string(info.frames[i].func),
                 exception_string(info.frames[i].file),
                 Int(info.frames[i].line)) for i in 1:nframes]
            # clear the flag, lock and payload so the mailbox is reusable (and other queues
            # checking it don't re-report the exception)
            unsafe_store!(ptr, ExceptionInfo_st())
            KernelException(queue.device, exception_string(info.name),
                            exception_string(info.reason),
                            Int.(info.work_item[1:3]), Int.(info.work_group[1:3]),
                            backtrace)
        end
        empty!(mailbox.pending)
        exc === nothing || throw(exc)
    end
    return
end
