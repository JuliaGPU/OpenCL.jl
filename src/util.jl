function create_compute_context()
    ctx    = create_some_context()
    device = first(devices(ctx))
    queue  = CmdQueue(ctx)
    return (device, ctx, queue)
end

opencl_version(obj :: CLObject) = api.parse_version(obj[:version])
opencl_version(c :: Context)  = opencl_version(first(devices(c)))
opencl_version(q :: CmdQueue) = opencl_version(q[:device])

const _versionDict = Dict{Ptr{Nothing}, VersionNumber}()

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
function format(s::String; vars...)
    for (k, v) in vars
        s = replace(s, "%($k)"=>v)
    end
    s
end

function build_kernel(ctx::Context, program::String,
                      kernel_name::String; vars...)
    src = format(program; vars...)
    p = Program(ctx, source=src)
    build!(p)
    return Kernel(p, kernel_name)
end

# cache for kernels; dict of form `(program_file, kernel_name, vars) -> kernel`
const CACHED_KERNELS = Dict{Tuple{String, String, Dict}, Kernel}()

function get_kernel(ctx::Context, program_file::String,
                    kernel_name::String; vars...)
    key = (program_file, kernel_name, Dict(vars))
    if in(key, keys(CACHED_KERNELS))
        return CACHED_KERNELS[key]
    else
        kernel = build_kernel(ctx, Base.read(program_file, String), kernel_name; vars...)
        CACHED_KERNELS[key] = kernel
        return kernel
    end
end

min_v11(elem) = check_version(elem, v"1.1")
min_v12(elem) = check_version(elem, v"1.2")
min_v20(elem) = check_version(elem, v"2.0")
min_v21(elem) = check_version(elem, v"2.1")
min_v22(elem) = check_version(elem, v"2.2")
