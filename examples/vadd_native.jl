using OpenCL, pocl_jll, Test

function vadd(a, b, c)
    i = get_global_id()
    @inbounds c[i] = a[i] + b[i]
    return
end

dims = (2,)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = CLArray(a)
d_b = CLArray(b)
d_c = CLArray(c)

len = prod(dims)
@opencl global_size=len vadd(d_a, d_b, d_c)
c = Array(d_c)
@test a+b ≈ c
