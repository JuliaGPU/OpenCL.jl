clbox{T}(x::T) = T[x]
unbox{T}(x::Array{T,1}) = x[1]

function create_compute_context()
    ctx    = create_some_context()
    device = first(devices(ctx))
    queue  = CmdQueue(ctx)
    return (device, ctx, queue)
end
