# GPUArrays.jl interface

function GPUArrays.derive(::Type{T}, a::CLArray, dims::Dims{N}, offset::Int) where {T,N}
    ref = copy(a.data)
    offset = (a.offset * Base.elsize(a)) ÷ sizeof(T) + offset
    CLArray{T,N}(ref, dims; offset)
end

const GLOBAL_RNGs = Dict{cl.Device,GPUArrays.RNG}()
function GPUArrays.default_rng(::Type{<:CLArray})
    dev = cl.device()
    get!(GLOBAL_RNGs, dev) do
        N = dev.max_work_group_size
        state = CLArray{NTuple{4, UInt32}}(undef, N)
        rng = GPUArrays.RNG(state)
        Random.seed!(rng)
        rng
    end
end
