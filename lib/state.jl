export platform, device, context, queue


## platform selection

function platform()
    get!(task_local_storage(), :CLPlatform) do
        platforms = cl.platforms()
        if isempty(platforms)
            throw(ArgumentError("No OpenCL platforms found"))
        end

        # prefer platforms that implement the full profile
        full_platform = findfirst(platforms) do platform
            info(platform, :profile) == "FULL_PROFILE"
        end
        isnothing(full_platform) || return platforms[full_platform]

        # otherwise, just return the first platform
        return first(cl.platforms())
    end::cl.Platform
end

# allow overriding with a specific platform
function platform!(p::cl.Platform)
    task_local_storage(:CLPlatform, p)
    delete!(task_local_storage(), :CLDevice)
    delete!(task_local_storage(), :CLContext)
    delete!(task_local_storage(), :CLQueue)
    return p
end

# allow selecting a platform by name or vendor
function platform!(name::String)
    platforms = cl.platforms()

    name_match = findfirst(platforms) do platform
        contains(lowercase(info(platform, :name)), lowercase(name))
    end
    isnothing(name_match) || return platform!(platforms[name_match])

    vendor_match = findfirst(platforms) do platform
        contains(lowercase(info(platform, :vendor)), lowercase(name))
    end
    isnothing(vendor_match) || return platform!(platforms[vendor_match])

    throw(ArgumentError("No OpenCL platform found with name or vendor $name"))
end


## device selection

function device()
    get!(task_local_storage(), :CLDevice) do
        dev = default_device(platform())
        isnothing(dev) && throw(ArgumentError("No OpenCL devices found"))
        dev
    end::cl.Device
end

# allow overriding with a specific device
function device!(dev::cl.Device)
    task_local_storage(:CLDevice, dev)
    delete!(task_local_storage(), :CLContext)
    delete!(task_local_storage(), :CLQueue)
    return dev
end

# allow selecting a device by type
function device!(dtype::Symbol)
    dev = cl.devices(platform(), dtype)
    isempty(dev) && throw(ArgumentError("No OpenCL devices found of type $dtype"))
    device!(first(dev))
end


## per-device contexts

# we use a single context per device
const context_lock = ReentrantLock()
const device_contexts = Dict{cl.Device, cl.Context}()
function context()
    get!(task_local_storage(), :CLContext) do
        @lock context_lock begin
            dev = device()
            get!(device_contexts, dev) do
                ctx = cl.Context(dev)
                device_contexts[dev] = ctx
                ctx
            end
        end
    end::cl.Context
end


## per-task queues

function queue()
    get!(task_local_storage(), :CLQueue) do
        q = cl.CmdQueue()
        task_local_storage(:CLQueue, q)
        q
    end::cl.CmdQueue
end
