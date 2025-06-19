# task-bound state

function clear_task_local_storage!()
    # the primary key for all task-local state is the context
    delete!(task_local_storage(), :CLContext)

    # all other state is derived
    delete!(task_local_storage(), :CLDevice)
    delete!(task_local_storage(), :CLPlatform)
    delete!(task_local_storage(), :CLQueue)
    delete!(task_local_storage(), :CLMemoryBackend)
end


## context creation

# we maintain a single global context per device
const device_contexts = Dict{Device, Context}()
const device_context_lock = ReentrantLock()
function device_context(dev::Device)
    return @lock device_context_lock begin
        get!(device_contexts, dev) do
            device_contexts[dev] = Context(dev)
        end
    end
end

function context()
    return get!(task_local_storage(), :CLContext) do
        dev = if haskey(task_local_storage(), :CLDevice)
            device()
        elseif haskey(task_local_storage(), :CLPlatform)
            default_device(platform())
        else
            default_device(default_platform())
        end
        isnothing(dev) && throw(ArgumentError("No OpenCL devices found"))
        device_context(dev)
    end::Context
end

function context!(ctx::Context)
    ctx == context() && return ctx

    clear_task_local_storage!()
    task_local_storage(:CLContext, ctx)
    return ctx
end

# temporarily switch the current context to a different context
function context!(f::Base.Callable, args...)
    old = context()
    context!(args...)
    try
        f()
    finally
        context!(old)
    end
end


## platform selection

function default_platform()
    ps = platforms()
    if isempty(ps)
        throw(ArgumentError("No OpenCL platforms found"))
    end

    # prefer platforms that implement the full profile
    idx = findfirst(ps) do p
        p.profile == "FULL_PROFILE"
    end
    isnothing(idx) || return ps[idx]

    # otherwise, just return the first platform
    return first(ps)
end

function platform()
    get!(task_local_storage(), :CLPlatform) do
        device().platform
    end::Platform
end

# allow overriding with a specific platform
function platform!(p::Platform)
    p == platform() && return p

    clear_task_local_storage!()
    task_local_storage(:CLPlatform, p)
    return p
end

# allow selecting a platform by name or vendor
function platform!(name::String)
    ps = platforms()

    # check the name
    idx = findfirst(ps) do p
        contains(lowercase(p.name), lowercase(name))
    end
    isnothing(idx) || return platform!(ps[idx])

    # check the vendor
    idx = findfirst(ps) do p
        contains(lowercase(p.vendor), lowercase(name))
    end
    isnothing(idx) || return platform!(ps[idx])

    throw(ArgumentError("No OpenCL platform found with name or vendor $name"))
end


## device selection

function default_device(p::Platform)
    devs = devices(p, CL_DEVICE_TYPE_DEFAULT)
    isempty(devs) && return nothing
    # XXX: clGetDeviceIDs documents CL_DEVICE_TYPE_DEFAULT should only return one device,
    #      but it's been observed to return multiple devices on some platforms...
    return first(devs)
end

function device()
    get!(task_local_storage(), :CLDevice) do
        only(context().devices)
    end::Device
end

# allow overriding with a specific device
function device!(dev::Device)
    dev == device() && return dev

    clear_task_local_storage!()
    task_local_storage(:CLDevice, dev)
    return dev
end

# allow selecting a device by type
function device!(dtype::Symbol)
    dev = devices(platform(), dtype)
    isempty(dev) && throw(ArgumentError("No OpenCL devices found of type $dtype"))
    device!(first(dev))
end

# temporarily switch the current device to a different device
function device!(f::Base.Callable, args...)
    old = device()
    device!(args...)
    try
        f()
    finally
        device!(old)
    end
end


## memory back-end

abstract type AbstractMemoryBackend end
struct SVMBackend <: AbstractMemoryBackend end
struct USMBackend <: AbstractMemoryBackend end
struct BDABackend <: AbstractMemoryBackend end

function supported_memory_backends(dev::Device)
    backends = AbstractMemoryBackend[]

    # unified shared memory is the first choice, as it gives us separate host and device
    # memory spaces that can be directly referenced by raw pointers.
    if usm_supported(dev)
        usm_caps = usm_capabilities(dev)
        if usm_caps.host.access && usm_caps.device.access
            push!(backends, USMBackend())
        end
    end

    # plain old device buffers are second choice, but require an extension to support being
    # referenced by raw pointers.
    if bda_supported(dev)
        push!(backends, BDABackend())
    end

    # shared virtual memory is last, because it comes at a performance cost.
    svm_caps = svm_capabilities(dev)
    if svm_caps.coarse_grain_buffer
        push!(backends, SVMBackend())
    end

    return backends
end

function default_memory_backend(dev::Device)
    supported_backends = supported_memory_backends(dev)
    isempty(supported_backends) && return nothing

    backend_str = load_preference(OpenCL, "default_memory_backend")
    backend_str === nothing && return first(supported_backends)

    backend = if backend_str == "usm"
        USMBackend()
    elseif backend_str == "bda"
        BDABackend()
    elseif backend_str == "svm"
        SVMBackend()
    else
        error("Unknown memory backend '$backend_str' requested")
    end
    in(backend, supported_backends) ? backend : nothing
    backend
end

function memory_backend()
    return get!(task_local_storage(), :CLMemoryBackend) do
        backend = default_memory_backend(device())
        if backend === nothing
            error("Device $(device()) does not support any of the available memory backends")
        end
        backend
    end
end


## per-task queues

function queue()
    get!(task_local_storage(), :CLQueue) do
        dev = device()

        # switching between devices on a task should yield the same queues
        queues = get!(task_local_storage(), :CLQueues) do
            Dict{Device, CmdQueue}()
        end

        get!(queues, dev) do
            CmdQueue()
        end
    end::CmdQueue
end

# switch the current task to a different queue
function queue!(q::CmdQueue)
    if q.device != device()
        throw(ArgumentError("Cannot switch to a queue on a different device"))
    end
    task_local_storage(:CLQueue, q)
    return q
end

# allow selecting a queue by properties
function queue!(args...)
    q = CmdQueue(args...)
    task_local_storage(:CLQueue, q)
    return q
end

# temporarily switch the current task to a different queue
function queue!(f::Base.Callable, args...)
    old = queue()
    queue!(args...)
    try
        f()
    finally
        queue!(old)
    end
end
