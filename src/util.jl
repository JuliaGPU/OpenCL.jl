function create_compute_context()
    ctx    = create_some_context()
    device = first(devices(ctx))
    queue  = CmdQueue(ctx)
    return (device, ctx, queue)
end

opencl_version(p :: Platform) = api.parse_version(p[:version])
opencl_version(d :: Device)   = api.parse_version(d[:version])
opencl_version(c :: Context)  = opencl_version(first(devices(c)))
opencl_version(q :: CmdQueue) = opencl_version(q[:device])

const _versionDict = Dict{Ptr{Void}, VersionNumber}()

_deletecached!(obj) = delete!(_versionDict, pointer(obj))

function check_version(obj, version :: VersionNumber)
    version <= get!(_versionDict, pointer(obj)) do
        opencl_version(obj)
    end
end
