# OpenCL.jl

[![][buildkite-img]][buildkite-url] [![][github-img]][github-url]

[buildkite-img]: https://badge.buildkite.com/6b2a46bff67692115dea3ad5a275d2f80777a5a99ffe42adb0.svg
[buildkite-url]: https://buildkite.com/julialang/opencl-dot-jl
[github-img]: https://github.com/JuliaGPU/OpenCL.jl/actions/workflows/CI.yml/badge.svg
[github-url]: https://github.com/JuliaGPU/OpenCL.jl/actions/workflows/CI.yml

*Julia interface for the OpenCL parallel computation API*

This package aims to be a complete solution for OpenCL programming in Julia, similar in
scope to [PyOpenCL] for Python. It provides a high level API for OpenCL to make programing
hardware accelerators, such as GPUs, FPGAs, and DSPs, as well as multicore CPUs much less
onerous.

**OpenCL.jl needs your help! If you can help maintaining this package, please reach out on
the [JuliaLang Slack](https://julialang.org/slack/) #gpu channel**

**Also note: OpenCL.jl is currently undergoing major changes. If you have old code,
developed for OpenCL.jl v0.9, please check [`NEWS.md`](NEWS.md) for an overview of the
changes.**


## Installation

1. Install an OpenCL driver. You can install one system-wide, i.e., using your package
   manager, or use `pocl_jll.jl` for a CPU back-end.
2. Add OpenCL to your Julia environment:

```julia
using Pkg
Pkg.add("OpenCL")
```


## Basic example: vector add

```julia
using LinearAlgebra
using OpenCL, pocl_jll

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

a_buff = cl.Buffer(Float32, length(a), (:r, :copy), hostbuf=a)
b_buff = cl.Buffer(Float32, length(b), (:r, :copy), hostbuf=b)
c_buff = cl.Buffer(Float32, length(a), :w)

p = cl.Program(source=sum_kernel) |> cl.build!
k = cl.Kernel(p, "sum")

cl.launch(k, size(a), nothing, a_buff, b_buff, c_buff)

r = cl.read(c_buff)

if isapprox(norm(r - (a+b)), zero(Float32))
    @info "Success!"
else
    @error "Norm should be 0.0f"
end
```


## More examples

You may want to check out the `examples` folder. Either `git clone` the repository to your
local machine or navigate to the OpenCL.jl install directory via:

```julia
using OpenCL
cd(joinpath(dirname(pathof(OpenCL)), ".."))
```

Otherwise, feel free to take a look at the Jupyter notebooks below:

  * [Julia set fractals](https://github.com/JuliaGPU/OpenCL.jl/blob/master/examples/notebooks/julia_set_fractal.ipynb)
  * [Mandlebrot fractal](https://github.com/JuliaGPU/OpenCL.jl/blob/master/examples/notebooks/mandelbrot_fractal.ipynb)
  * [Transpose bandwidth](https://github.com/JuliaGPU/OpenCL.jl/blob/master/examples/notebooks/Transpose.ipynb)


## Credit

This package is heavily influenced by the work of others:

  * [PyOpenCL](http://mathema.tician.de/software/pyopencl/) by Andreas Klockner
  * [oclpb](https://github.com/srossross/oclpb) by Sean Ross
  * [Boost.Compute](https://github.com/kylelutz/compute) by Kyle Lutz
  * [rust-opencl](https://github.com/luqmana/rust-opencl)
