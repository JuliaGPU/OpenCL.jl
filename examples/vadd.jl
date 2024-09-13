using OpenCL, pocl_jll, Test

const source = """
   __kernel void vadd(__global const float *a,
                      __global const float *b,
                      __global float *c) {
      int i = get_global_id(0);
      c[i] = a[i] + b[i];
    }"""

dims = (2,)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = CLArray(a)
d_b = CLArray(b)
d_c = CLArray(c)

prog = cl.Program(; source) |> cl.build!
kern = cl.Kernel(prog, "vadd")

len = prod(dims)
clcall(kern, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}},
       d_a, d_b, d_c; global_size=(len,))
c = Array(d_c)
@test a+b ≈ c
