lbox{T}(x::T) = T[x]
unbox{T}(x::Array{T,1}) = x[1]

function create_compute_context()
    ctx    = create_some_context()
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
    return (int(mg.captures[1]), int(mg.captures[2]))
end 

function opencl_version(d::Device)
    return opencl_version(d[:platform])
end

function opencl_version(c::Context)
    return opencl_version(first(devices(c)))
end

function opencl_version(q::CmdQueue)
    return opencl_version(q[:context])
end
