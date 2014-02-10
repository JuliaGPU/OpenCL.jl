using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

function twiddle(u::Uint32, v::Uint32)
    t1 = ((u & 0x80000000) | (v & 0x7FFFFFFF)) >> int32(1)
    local t2::Uint32
    if v & uint32(1) == zero(Uint32)
        t2 = uint32(0x0)
    else
        t2 = uint32(0x9908B0DF)
    end
    return t1 $ t2
end

@clkernel generate_state(state::Vector{Uint32}) = begin
    n = uint32(624)
    m = uint32(397)
    for i = int32(0:(n - m - 1))
        state[i] = state[i] + m
    end
    for i = int32(0:(n - m - 1))
        state[i] = state[i+m] $ twiddle(state[i], state[i+1])
    end
    for i = int32((n - m):(n - 2))
        state[i] = state[i+m-n] $ twiddle(state[i], state[i+1])
    end
    state[n-1] = state[m-1] $ twiddle(state[n-1], state[0])
    return
end

function generate_state2(state::Vector{Uint32})
    n = uint32(624)
    m = uint32(397)
    for i = int32(0:(n - m - 1))
        state[i] = state[i] + m
    end
    for i = int32(0:(n - m - 1))
       state[i] = state[i+m] $ twiddle(state[i], state[i+1])
    end
    for i = int32((n - m):(n - 2))
        state[i] = state[i+m-n] $ twiddle(state[i], state[i+1])
    end
    state[n-1] = state[m-1] $ twiddle(state[n-1], state[0])
    return
end

@clkernel seed(s::Uint32, state::Vector{Uint32}) = begin
    n = uint32(624)
    m = uint32(397)
    state[0] = s & 0xFFFFFFFF
    for i = int32(1:(n-1))
        state[i] = 1812433253 * (state[i-1] $ (state[i-1] >> int32(30))) + uint32(i)
        state[i] = state[i] & 0xFFFFFFFF
    end
    return generate_state2(state)
end

@assert isa(seed, cl.Kernel)

function random_number(state::Vector{Uint32}, p::Cuint)
    x = state[p]
    x $= (x >> int32(11))
    x $= (x << int32(7)) & 0x9D2C5680
    x $= (x << int32(15)) & 0xEFC60000
    return x $ (x >> int32(8))
end
    
@clkernel fill(state::Vector{Uint32}, vector::Vector{Uint32}, offset::Cuint) = begin
    i = get_global_id(0)
    vector[offset + i] = random_number(state, i)
    return
end

@assert isa(fill, cl.Kernel)

function seed_mersenne!{T}(state_buffer::cl.Buffer{T})
    n = length(state_buffer)
    seed[queue, (n,)](uint32(n), state_buffer)
    return
end

function test_fill!{T}(b::Vector{T}, seed_kernel, fill_kernel, generate_state_kernel)
    n = 624
    m = 397 
    len = length(b)

    buffer = cl.Buffer(T, ctx, (:rw, :copy), hostbuf=b)
    
    state_buffer = cl.Buffer(T, ctx, :rw, n)
    seed_kernel[queue, (n,)](uint32(n), state_buffer)
    
    cl.set_arg!(fill_kernel, 1, state_buffer)
    cl.set_arg!(fill_kernel, 2, buffer)
    
    p = 0
    while true
        cnt = 0
        if len - p >= n
            cnt = n
        else
            cnt = len - p
        end
        cl.set_arg!(fill_kernel, 3, uint32(p))
        cl.enqueue_kernel(queue, fill_kernel, (cnt,))
        p += n
        if p >= len
            break
        end
        cl.set_arg!(generate_state_kernel, 1, state_buffer)
        cl.enqueue_kernel(queue, generate_state_kernel, (1,), (1,))
    end
    return cl.read(queue, buffer)
end

facts("Test example generate mersenne") do
    z = zeros(Float32, 1000_000)
    rjulia = test_fill!(z, seed, fill, generate_state)

    src = open(readall, "mersenne_twister.cl")
    prg = cl.Program(ctx, source=src) |> cl.build!
    
    generate_state_comp = cl.Kernel(prg, "generate_state")
    seed_comp = cl.Kernel(prg, "seed")
    fill_comp = cl.Kernel(prg, "fill")
   
    z = zeros(Float32, 1000_000)
    rocl  = test_fill!(z, seed_comp, fill_comp, generate_state_comp)
    
    delta  = abs(rjulia - rocl)
    @fact isapprox(sum(delta), 0.0f0) => true
end
