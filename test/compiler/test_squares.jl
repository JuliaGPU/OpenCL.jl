using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

@clkernel squared(in::Vector{Float32},
                  out::Vector{Float32}) = begin
    gid = get_global_id(0)
    out[gid] = in[gid] * in[gid]
    return
end

facts("Test example squares") do
    n = 1_000_000
    in  = cl.Buffer(Float32, ctx, n)
    out = cl.Buffer(Float32, ctx, n)
    
    cl.fill!(queue, in, 2.0f0)

    squared[queue, (n,)](in, out)
    res = cl.read(queue, out)

    @fact all(x -> x == 4.0f0, res) => true
end
