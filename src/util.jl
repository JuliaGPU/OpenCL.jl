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

"""
Format string using dict-like variables, replacing all accurancies of
`%(key)` with `value`.

Example:
    s = "Hello, %(name)"
    format(s, name="Tom")  ==> "Hello, Tom"
"""
function format(s::AbstractString; vars...)
    for (k, v) in vars
        s = replace(s, "%($k)", v)
    end
    s
end

function build_kernel(ctx::Context, program::AbstractString,
                      kernel_name::AbstractString; vars...)
    src = format(program; vars...)
    p = Program(ctx, source=src)
    build!(p)
    return Kernel(p, kernel_name)
end

# cache for kernels; dict of form `(program_file, kernel_name, vars) -> kernel`
@compat const CACHED_KERNELS = Dict{Tuple{AbstractString, AbstractString, Dict}, Kernel}()

function get_kernel(ctx::Context, program_file::AbstractString,
                    kernel_name::AbstractString; vars...)
    key = (program_file, kernel_name, Dict(vars))
    if in(key, keys(CACHED_KERNELS))
        return CACHED_KERNELS[key]
    else
        kernel = build_kernel(ctx, readall(program_file), kernel_name; vars...)
        CACHED_KERNELS[key] = kernel
        return kernel       
    end
end

