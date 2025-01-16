## platform selection

function platform()
    return get!(task_local_storage(), :CLPlatform) do
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
    delete!(task_local_storage(), :CLContext)
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
    return get!(task_local_storage(), :CLDevice) do
        dev = default_device(platform())
        isnothing(dev) && throw(ArgumentError("No OpenCL devices found"))
        dev
    end::Device
end

# allow overriding with a specific device
function device!(dev::Device)
    task_local_storage(:CLDevice, dev)
    delete!(task_local_storage(), :CLContext)
    delete!(task_local_storage(), :CLQueue)
    return dev
end

# allow selecting a device by type
function device!(dtype::Symbol)
    dev = devices(platform(), dtype)
    isempty(dev) && throw(ArgumentError("No OpenCL devices found of type $dtype"))
    return device!(first(dev))
end


## per-device contexts

# we use a single context per device
const context_lock = ReentrantLock()
const device_contexts = Dict{Device, Context}()
function context()
    return get!(task_local_storage(), :CLContext) do
        @lock context_lock begin
            dev = device()
            get!(device_contexts, dev) do
                ctx = Context(dev)
                device_contexts[dev] = ctx
                ctx
            end
        end
    end::Context
end

# temporarily switch the current device to a different device
function device!(f::Base.Callable, args...)
    old = device()
    device!(args...)
    return try
        f()
    finally
        device!(old)
    end
end


## per-task queues

# XXX: port CUDA.jl's per-array stream tracking, obviating the need for global sync
const queues = WeakKeyDict{cl.CmdQueue, Nothing}()
function device_synchronize()
    for queue in keys(queues)
        cl.finish(queue)
    end
    return
end

function queue()
    return get!(task_local_storage(), :CLQueue) do
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
    return try
        f()
    finally
        queue!(old)
    end
end

############### Taken from CUDA.jl ##################
#=
@enum MathMode begin
    # use prescribed precision and standardized arithmetic for all calculations.
    # this may serialize operations, and reduce performance.
    PEDANTIC_MATH

    # use at least the required precision, and allow reordering operations for performance.
    DEFAULT_MATH

    # additionally allow downcasting operations for better use of hardware resources.
    # whenever possible the `precision` flag passed to `math_mode!` will be used
    # to constrain those downcasts.
    FAST_MATH
end

# math mode and precision are sticky (once set on a task, inherit to newly created tasks)
const default_math_mode = Ref{Union{Nothing,MathMode}}(nothing)
const default_math_precision = Ref{Union{Nothing,Symbol}}(nothing)

# the default device unitialized tasks will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
# this setting won't be used when switching tasks on a pre-initialized thread.
# const default_device = Ref{Union{Nothing,Device}}(nothing)


mutable struct TaskLocalState
    device::Device
    context::Context
    queues::Vector{Union{Nothing,CmdQueue}}
    math_mode::MathMode
    math_precision::Symbol

    function TaskLocalState(dev::Device=something(default_device(platform()), Device(0)),
                            ctx::Context = context(dev))
        math_mode = something(default_math_mode[],
                              Base.JLOptions().fast_math==1 ? FAST_MATH : DEFAULT_MATH)
        math_precision = something(default_math_precision[], :TensorFloat32)
        new(dev, ctx, Union{Nothing,CmdQueue}[nothing for _ in 1:length(device())],
            math_mode, math_precision)
    end
end

function validate_task_local_state(state::TaskLocalState)
    # NOTE: the context may be invalid if another task reset it (which we detect here
    #       since we can't touch other tasks' local state from `device_reset!`)
    if !isvalid(state.context)
        device!(state.device)
        @inbounds state.queues[state.device.id+1] = nothing
    end
    return state
end

# get or create the task local state, and make sure it's valid
function task_local_state!(args...)
    tls = task_local_storage()
    if haskey(tls, :OpenCL)
        validate_task_local_state(@inbounds(tls[:OpenCL])::TaskLocalState)
    else
        # verify that CUDA.jl is functional. this doesn't belong here, but since we can't
        # error during `__init__`, we do it here instead as this is the first function
        # that's likely executed when using CUDA.jl
        # TODO: Check if something similar exists or needs to be done for OpenCL
        # @assert functional(true)

        tls[:OpenCL] = TaskLocalState(args...)
    end::TaskLocalState
end

# only get the task local state (it may be invalid!), or return nothing if unitialized
function task_local_state()
    tls = task_local_storage()
    if haskey(tls, :OpenCL)
        @inbounds(tls[:OpenCL])
    else
        nothing
    end::Union{TaskLocalState,Nothing}
end

@noinline function create_stream()
    stream = CmdQueue()

    # register the name of this task
    # XXX: do this when the user has imported NVTX.jl (using weak dependencies?)
    #t = current_task()
    #tptr = pointer_from_objref(current_task())
    #tptrstr = string(convert(UInt, tptr), base=16, pad=Sys.WORD_SIZE>>2)
    #NVTX.nvtxNameCuStreamA(stream, "Task(0x$tptrstr)")

    stream
end


@inline function queue(state=task_local_state!())
    # @inline so that it can be DCE'd when unused from active_state
    devidx = state.device.id
    @inbounds if state.queues[devidx] === nothing
        state.streams[devidx] = create_stream()
    else
        state.queues[devidx]::CmdQueue
    end
end


@inline function active_state()
    # inline to remove unused state properties
    state = task_local_state!()
    return (device=state.device, context=state.context, queue=queue(state),
            math_mode=state.math_mode, math_precision=state.math_precision)
end
=#
