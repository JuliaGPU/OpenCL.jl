export opencl_version

function parse_version(version_string)
    mg = match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
    if mg === nothing
        error("Non conforming version string: $(ver)")
    end
    return VersionNumber(parse(Int, mg.captures[1]),
                                 parse(Int, mg.captures[2]))
end

opencl_version(obj::CLObject) = parse_version(obj.version)
opencl_version(c::cl.Context)  = opencl_version(first(c.devices))
opencl_version(q::cl.CmdQueue) = opencl_version(q.device)

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

function build_kernel(program::String, kernel_name::String; vars...)
    src = format(program; vars...)
    p = cl.Program(source=src)
    cl.build!(p)
    return cl.Kernel(p, kernel_name)
end

const CACHED_KERNELS = Dict{Any, cl.Kernel}()
function get_kernel(program_file::String, kernel_name::String; vars...)
    key = (cl.context(), program_file, kernel_name, Dict(vars))
    if in(key, keys(CACHED_KERNELS))
        return CACHED_KERNELS[key]
    else
        kernel = build_kernel(read(program_file, String), kernel_name; vars...)
        CACHED_KERNELS[key] = kernel
        return kernel
    end
end

function versioninfo(io::IO=stdout)
    println(io, "OpenCL.jl version $(pkgversion(@__MODULE__))")
    println(io)

    println(io, "Toolchain:")
    println(io, " - Julia v$(VERSION)")
    pkgs = []
    if !Sys.isapple()
        push!(pkgs, OpenCL_jll)
    end
    for pkg in pkgs
        println(io, " - $(string(pkg)) v$(pkgversion(pkg))")
    end
    println(io)

    env = filter(var->startswith(var, "JULIA_OPENCL"), keys(ENV))
    if !isempty(env)
        println(io, "Environment:")
        for var in env
            println(io, "- $var: $(ENV[var])")
        end
        println(io)
    end

    println(io, "Available platforms: ", length(cl.platforms()))
    for platform in cl.platforms()
        println(io, " - $(platform.name)")
        println(io, "   version: $(platform.version)")
        for device in cl.devices(platform)
            print(io, "   · $(device.name)")

            ## list some relevant extensions
            extensions = []
            if in("cl_khr_fp16", device.extensions)
                push!(extensions, "fp16")
            end
            if in("cl_khr_fp64", device.extensions)
                push!(extensions, "fp64")
            end
            if in("cl_khr_il_program", device.extensions)
                push!(extensions, "il")
            end
            if !isempty(extensions)
                print(io, " (", join(extensions, ", "), ")")
            end
            println(io)
        end
    end
end
