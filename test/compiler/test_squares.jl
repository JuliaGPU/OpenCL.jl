using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

@clkernel squared(a::Vector{Float32},
                  b::Vector{Float32},
                  c::Vector{Float32}) = begin
    gid = get_global_id(0)
    c[gid] = a[gid] * b[gid]
    return
end

facts("Test example squares") do
    n = 1_000_000
    a = cl.Buffer(Float32, ctx, n)
    b = cl.Buffer(Float32, ctx, n)
    c = cl.Buffer(Float32, ctx, n)

    cl.fill!(queue, a, 2.0f0)
    cl.fill!(queue, b, 2.0f0)

    squared[queue, (n,)](a, b, c)
    res = cl.read(queue, c)

    @fact all(x -> x == 4.0f0, res) => true
end
