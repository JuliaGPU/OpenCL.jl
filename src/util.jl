function create_compute_context()
    ctx    = create_some_context()
    device = first(devices(ctx))
    queue  = CmdQueue(ctx)
    return (device, ctx, queue)
end

opencl_version(obj :: CLObject) = api.parse_version(obj[:version])
opencl_version(c :: Context)  = opencl_version(first(devices(c)))
opencl_version(q :: CmdQueue) = opencl_version(q[:device])

const _versionDict = Dict{Ptr{Void}, VersionNumber}()

_deletecached!(obj :: CLObject) = delete!(_versionDict, pointer(obj))

function check_version(obj :: CLObject, version :: VersionNumber)
    version <= get!(_versionDict, pointer(obj)) do
        opencl_version(obj)
    end
end
