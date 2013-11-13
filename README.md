# OpenCL.jl

OpenCL 1.2 bindings for Julia

[![Build Status](https://travis-ci.org/jakebolewski/OpenCL.jl.png)](https://travis-ci.org/jakebolewski/OpenCL.jl)

Julia interface for the OpenCL parallel computation API

This package aims to be a complete solution for OpenCL programming in Julia, similar in scope to [PyOpenCL] for Python.
It provides a high level api for OpenCL to make programing GPU's and multicore CPU's much less onerous.

#### This package is based off the work of others:
  * [PyOpenCL] by Andreas Klockner
  * [oclpb]    by Sean Ross
  * [Boost.Compute] by Kyle Lutz
  * [rust-opencl]

[PyOpenCL]: http://mathema.tician.de/software/pyopencl/
[oclpb]: https://github.com/srossross/oclpb
[Boost.Compute]:https://github.com/kylelutz/compute
[rust-opencl]: https://github.com/luqmana/rust-opencl

## Example Notebooks
  * [OpenCL Fractals]
  * [GPU Buffer Transpose]
  * [Low Level API]

[OpenCL Fractals]:http://nbviewer.ipython.org/7436359
[GPU Buffer Transpose]:http://nbviewer.ipython.org/7436439
[Low Level API]:http://nbviewer.ipython.org/7452048

## Quick Example

```julia

import OpenCL
const cl = OpenCL

const sum_kernel = "
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
c_buff = cl.Buffer(Float32, ctx, :w, sizeof(a))

p = cl.Program(ctx, source=sum_kernel) |> cl.build!
k = cl.Kernel(p, "sum")

cl.call(queue, k, size(a), nothing, a_buff, b_buff, c_buff)

r = cl.read(queue, c_buff)

if isapprox(norm(r - (a+b)), zero(Float32))
    info("Success!")
else
    error("Norm should be 0.0f")
end
```
