using OpenCL

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

# create opencl buffer objects
# copies to the device initiated when the kernel function is called
a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=a)
b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=b)
c_buff = cl.Buffer(Float32, ctx, :w, length(a))

# build the program and construct a kernel object
p = cl.Program(ctx, source=sum_kernel_src) |> cl.build!
sum_kernel = cl.Kernel(p, "sum")

# call the kernel object with global size set to the size our arrays
sum_kernel[queue, size(a)](a_buff, b_buff, c_buff)

# perform a blocking read of the result from the device
r = cl.read(queue, c_buff)

# check to see if our result is what we expect!
if isapprox(norm(r - (a+b)), zero(Float32))
    info("Success!")
else
    error("Norm should be 0.0f")
end
