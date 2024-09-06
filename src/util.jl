export create_compute_context, opencl_version

function create_compute_context()
    ctx    = create_some_context()
    device = first(devices(ctx))
    queue  = cl.CmdQueue(ctx)
    return (device, ctx, queue)
end

function parse_version(version_string)
    mg = match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
    if mg === nothing
        error("Non conforming version string: $(ver)")
    end
    return VersionNumber(parse(Int, mg.captures[1]),
                                 parse(Int, mg.captures[2]))
end

opencl_version(obj::CLObject) = parse_version(obj[:version])
opencl_version(c::cl.Context)  = opencl_version(first(cl.devices(c)))
opencl_version(q::cl.CmdQueue) = opencl_version(q[:device])

"""
Format string using dict-like variables, replacing all accurancies of
`%(key)` with `value`.

Example:
    s = "Hello, %(name)"
    format(s, name="Tom")  ==> "Hello, Tom"
"""
function format(s::String; vars...)
    for (k, v) in vars
        s = replace(s, "%($k)" => v)
    end
    s
end

function build_kernel(ctx::cl.Context, program::String,
                      kernel_name::String; vars...)
    src = format(program; vars...)
    p = cl.Program(ctx, source=src)
    cl.build!(p)
    return cl.Kernel(p, kernel_name)
end

# cache for kernels; dict of form `(program_file, kernel_name, vars) -> kernel`
const CACHED_KERNELS = Dict{Tuple{String, String, Dict}, cl.Kernel}()

function get_kernel(ctx::cl.Context, program_file::String,
                    kernel_name::String; vars...)
    key = (program_file, kernel_name, Dict(vars))
    if in(key, keys(CACHED_KERNELS))
        return CACHED_KERNELS[key]
    else
        kernel = build_kernel(ctx, read(program_file, String), kernel_name; vars...)
        CACHED_KERNELS[key] = kernel
        return kernel
    end
end
