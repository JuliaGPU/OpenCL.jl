lbox{T}(x::T) = T[x]
unbox{T}(x::Array{T,1}) = x[1]

function create_compute_context(dtype=:all)
    ctx    = create_some_context(dtype)
    device = first(devices(ctx))
    queue  = CmdQueue(ctx)
    return (device, ctx, queue)
end

function opencl_version(p::Platform)
    ver = p[:version]
    mg = match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", ver) 
    if mg == nothing
        error("Platform $(p[:name]) return non conformat platform string: $(ver)")
    end
    return VersionNumber(int(mg.captures[1]), int(mg.captures[2]))
end 

opencl_version(d::Device)   = opencl_version(d[:platform])
opencl_version(c::Context)  = opencl_version(first(devices(c)))
opencl_version(q::CmdQueue) = opencl_version(q[:context])
