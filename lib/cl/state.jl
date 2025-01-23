## platform selection

function platform()
    get!(task_local_storage(), :CLPlatform) do
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
    end::Platform
end

# allow overriding with a specific platform
function platform!(p::Platform)
    task_local_storage(:CLPlatform, p)
    delete!(task_local_storage(), :CLDevice)
    delete!(task_local_storage(), :CLDeviceState)
    delete!(task_local_storage(), :CLQueue)
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

function device()
    get!(task_local_storage(), :CLDevice) do
        dev = default_device(platform())
        isnothing(dev) && throw(ArgumentError("No OpenCL devices found"))
        dev
    end::Device
end

# allow overriding with a specific device
function device!(dev::Device)
    task_local_storage(:CLDevice, dev)
    task_local_storage(:CLPlatform, dev.platform)
    delete!(task_local_storage(), :CLDeviceState)
    delete!(task_local_storage(), :CLQueue)
    return dev
end

# allow selecting a device by type
function device!(dtype::Symbol)
    dev = devices(platform(), dtype)
    isempty(dev) && throw(ArgumentError("No OpenCL devices found of type $dtype"))
    device!(first(dev))
end


## per-device state

# each device is associated with a single context
# (and some other state we only want to set up once)

abstract type AbstractMemoryBackend end
struct SVMBackend <: AbstractMemoryBackend end
struct USMBackend <: AbstractMemoryBackend end

struct DeviceState
    context::Context
    backend::AbstractMemoryBackend
end

const device_states = Dict{Device, DeviceState}()
const device_state_lock = ReentrantLock()
function device_state(dev::Device = device())
    return get!(task_local_storage(), :CLDeviceState) do
        @lock device_state_lock begin
            get!(device_states, dev) do
                ctx = Context(dev)

                # validate memory support

                # determine if USM is supported
                usm = if usm_supported(dev)
                    caps = usm_capabilities(dev)
                    caps.host.access && caps.device.access
                else
                    false
                end

                # determine if SVM is available (if needed)
                if !usm
                    caps = svm_capabilities(dev)
                    if !caps.coarse_grain_buffer
                        error("Device $dev does not support USM or coarse-grained SVM, either of which is required by OpenCL.jl")
                    end
                end

                backend = usm ? USMBackend() : SVMBackend()
                device_states[dev] = DeviceState(ctx, backend)
            end
        end
    end::DeviceState
end

context(dev::Device = device()) = device_state(dev).context
memory_backend(dev::Device = device()) = device_state(dev).backend

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


## per-task queues

# XXX: port CUDA.jl's per-array stream tracking, obviating the need for global sync
const queues = WeakKeyDict{cl.CmdQueue,Nothing}()
function device_synchronize()
    for queue in keys(queues)
        cl.finish(queue)
    end
end

function queue()
    get!(task_local_storage(), :CLQueue) do
        q = CmdQueue()
        task_local_storage(:CLQueue, q)
        queues[q] = nothing
        q
    end::CmdQueue
end

# switch the current task to a different queue
function queue!(q::CmdQueue)
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
