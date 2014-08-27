function create_compute_context()
    ctx    = create_some_context()
    device = first(devices(ctx))
    queue  = CmdQueue(ctx)
    return (device, ctx, queue)
end

opencl_version(p::Platform) = api.parse_version(p[:version])
opencl_version(d::Device)   = opencl_version(d[:platform])
opencl_version(c::Context)  = opencl_version(first(devices(c)))
opencl_version(q::CmdQueue) = opencl_version(q[:context])
