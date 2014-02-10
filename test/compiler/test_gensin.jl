using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

@clkernel generate_sin(a::Vector{Float32}, 
                       b::Vector{Float32}) = begin
    gid = get_global_id(0)
    n   = get_global_size(0)
    
    r = float32(gid) / float32(n)
    # sin wave with 8 oscillations
    y = r * (16.0f0 * 2.1415f0)
    # x is a range from -1 to 1
    a[gid] = r * 2.0f0 - 1.0f0
    # y is a sin wave
    b[gid] = sin(y)
    return
end

# under emulation how to deal 1 indexing? 
function generate_sin_julia(a::Vector{Float32}, b::Vector{Float32})
    n = length(a)
    for gid in 1:n
        r = (float32(gid) - 1) / float32(n)
        # sin wave with 8 oscillations
        y = r * (16.0f0 * 2.1415f0)
        # x is a range from -1 to 1
        a[gid] = r * 2.0f0 - 1.0f0
        # y is a sin wave
        b[gid] = sin(y)
    end
    return deepcopy(b)
end

facts("Test example generate sin wave") do
    n = 1_000_000
    a = cl.Buffer(Float32, ctx, n)
    b = cl.Buffer(Float32, ctx, n)

    evt  = generate_sin[queue, (n,)](a, b)
    rocl = cl.read(queue, b)

    a = Array(Float32, n)
    b = Array(Float32, n)
    rjulia = generate_sin_julia(a, b)

    delta  = abs(rjulia - rocl)
    l1norm = sum(delta) / sum(abs(rocl))
    info("L1 norm (OpenCL): $l1norm")
    @fact l1norm < 1.0e-7  => true
end
