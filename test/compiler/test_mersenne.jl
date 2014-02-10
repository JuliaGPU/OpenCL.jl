using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.CLAst

using OpenCL.SourceGen
using OpenCL.Compiler

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

function generate_state_julia(state::Vector{Uint32})
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
    x $= (x >>> int32(11))
    x $= (x <<< int32(7)) & 0x9D2C5680
    x $= (x <<< int32(15)) & 0xEFC60000
    return x $ (x >>> int32(8))
end
    
@clkernel fill(state::Vector{Uint32},
               vector::Vector{Uint32},
               offset::Cuint) = begin
    i = get_global_id(0)
    vector[offset + i] = random_number(state, i)
    return
end

@assert isa(fill, cl.Kernel)


#src = open(readall, "test.cl")
#prg = cl.Program(ctx, source=src) |> cl.build!
#generate_state = cl.Kernel(prg, "generate_state")
#seed = cl.Kernel(prg, "seed")
#fill = cl.Kernel(prg, "fill")

function seed_mersenne!{T}(state_buffer::cl.Buffer{T})
    n = length(state_buffer)
    cl.call(queue, seed, n, nothing, uint32(n), state_buffer)
    return
end

#seed_mersenne(Float32)

function test_fill{T}(b::Vector{T})
    n = 624
    m = 397 
    len = length(b)

    buffer = cl.Buffer(T, ctx, (:rw, :copy), hostbuf=b)
    
    state_buffer = cl.Buffer(T, ctx, :rw, n)
    seed_mersenne!(state_buffer)
    
    cl.set_arg!(fill, 1, state_buffer)
    cl.set_arg!(fill, 2, buffer)
    
    p = 0
    while true
        cnt = 0
        if len - p >= n
            cnt = n
        else
            cnt = len - p
        end
        cl.set_arg!(fill, 3, uint32(p))
        cl.enqueue_kernel(queue, fill, (cnt,))
        p += n
        if p >= len
            break
        end
        cl.set_arg!(generate_state, 1, state_buffer)
        cl.enqueue_kernel(queue, generate_state, (1,), (1,))
    end
    return cl.read(queue, buffer)
end

#z = zeros(Float32, 1_000_000)
#@time rand(Float32, 1_000_000)
#for _ = 1:10
#    @time test_fill(z)
#end

#@show test_fill(z)[1:50]


