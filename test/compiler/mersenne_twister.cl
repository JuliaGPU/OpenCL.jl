typedef struct Range {long start; long step; long len;} Range;

uint twiddle(uint u, uint v)
{
    return (((u & 0x80000000U) | (v & 0x7FFFFFFFU)) >> 1) ^ ((v & 1U) ? 0x9908B0DFU : 0x0U);
}

__kernel void generate_state(__global uint *state)
{ 
	const uint n = 624;
    const uint m = 397;
    for(int i = 0; i < (n - m); i++)
        state[i] = state[i+m] ^ twiddle(state[i], state[i+1]);
    for(int i = n - m; i < (n - 1); i++)
        state[i] = state[i+m-n] ^ twiddle(state[i], state[i+1]);
    state[n-1] = state[m-1] ^ twiddle(state[n-1], state[0]);
}

__kernel void seed(const uint s, __global uint *state)
{
    const uint n = 624;
    const uint m = 397;
    state[0] = s & 0xFFFFFFFFU;
    for(uint i = 1; i < n; i++){
        state[i] = 1812433253U * (state[i-1] ^ (state[i-1] >> 30)) + i;
        state[i] &= 0xFFFFFFFFU;
    }
    generate_state(state);
}

uint random_number(__global uint *state, const uint p)
{
    uint x = state[p];
    x ^= (x >> 11);
    x ^= (x << 7) & 0x9D2C5680U;
    x ^= (x << 15) & 0xEFC60000U;
    return x ^ (x >> 18);
}

__kernel void fill(__global uint *state,
                   __global uint *vector,
                   const uint offset)
{
    const uint i = get_global_id(0);
    vector[offset+i] = random_number(state, i);
}
