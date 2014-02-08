import OpenCL
const cl = OpenCL

const sum_kernel_src = "
   __kernel void sum(__global const float *a,
                     __global const float *b, 
                     __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
"
a = rand(Float32, 50_000)
b = rand(Float32, 50_000)

device, ctx, queue = cl.create_compute_context()

a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=a)
b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=b)
c_buff = cl.Buffer(Float32, ctx, :w, length(a))

p = cl.Program(ctx, source=sum_kernel_src) |> cl.build!
#k = cl.Kernel(p, "sum")

sum_kernel = cl.Kernel(p, "sum")
sum_kernel[queue, size(a)](a_buff, b_buff, c_buff)

#cl.call(queue, k, size(a), nothing, a_buff, b_buff, c_buff)

r = cl.read(queue, c_buff)

if isapprox(norm(r - (a+b)), zero(Float32))
    info("Success!")
else
    error("Norm should be 0.0f")
end
