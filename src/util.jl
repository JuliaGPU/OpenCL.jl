clbox{T}(x::T) = T[x]
unbox{T}(x::Array{T,1}) = x[1]

function create_compute_context()
    ctx   = create_some_context()
    queue = cl.CmdQueue(ctx)
    return (ctx, queue)
end


