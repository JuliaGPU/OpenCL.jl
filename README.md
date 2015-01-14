# OpenCL.jl

OpenCL bindings for Julia

[![Build Status](https://travis-ci.org/JuliaGPU/OpenCL.jl.svg?branch=master)](https://travis-ci.org/JuliaGPU/OpenCL.jl?branch=master)
[![Coverage Status](https://img.shields.io/coveralls/JuliaGPU/OpenCL.jl.png)](https://coveralls.io/r/JuliaGPU/OpenCL.jl?branch=master)

Julia interface for the OpenCL parallel computation API

This package aims to be a complete solution for OpenCL programming in Julia, similar in scope to [PyOpenCL] for Python.
It provides a high level api for OpenCL to make programing GPU's and multicore CPU's much less onerous.

OpenCL.jl provides access to OpenCL API versions 1.0, 1.1, 1.2 and 2.0.

#### Support of Julia v0.4
Currently `OpenCL.jl` only supports Julia v0.3 due to some breaking changes in Julia v0.4. Support is comming as soon as Julia v0.4 is entering its prerelease phase.

#### This package is based off the work of others:
  * [PyOpenCL] by Andreas Klockner
  * [oclpb]    by Sean Ross
  * [Boost.Compute] by Kyle Lutz
  * [rust-opencl]

[PyOpenCL]: http://mathema.tician.de/software/pyopencl/
[oclpb]: https://github.com/srossross/oclpb
[Boost.Compute]:https://github.com/kylelutz/compute
[rust-opencl]: https://github.com/luqmana/rust-opencl

#### Contributors
 * Jake Bolewski (@jakebolewski)
 * Valentin Churavy (@vchuravy)
 * Simon Danisch (@SimonDanisch)


## Setup
1. Install an OpenCL driver. If you use OSX, OpenCL is already available
2. Checkout the packages from the Julia repl

     ```julia
     Pkg.add("OpenCL")
     ```

3. OpenCL will be installed in your ``.julia`` directory
4. ``cd`` into your ``.julia`` directory to run the tests and try out the examples
5. To update to the latest development version, from the Julia repl:

      ```julia
         Pkg.update()
      ```

## IJulia Notebooks
  * [OpenCL Fractals]
  * [GPU Buffer Transpose]
  * [Low Level API]

[OpenCL Fractals]:http://nbviewer.ipython.org/7517923
[GPU Buffer Transpose]:http://nbviewer.ipython.org/7517952
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
c_buff = cl.Buffer(Float32, ctx, :w, length(a))

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
