# Profiling

OpenCL.jl applications can be profiled using [opencl-kernel-profiler](https://github.com/rjodinchr/opencl-kernel-profiler), which is based on Perfetto.
The most convenient way to use it is through `opencl_kernel_profiler_jll` by setting the `OPENCL_LAYERS` environment variable before initializing OpenCL as follows:

```julia
using opencl_kernel_profiler_jll
ENV["OPENCL_LAYERS"] = opencl_kernel_profiler_jll.libopencl_kernel_profiler
```

By default, traces are limited to 1024 KB. To increase this limit, set `CLKP_TRACE_MAX_SIZE` to a larger value, e.g. `ENV["CLKP_TRACE_MAX_SIZE"] = "100 * 1024"` for 100 MB.

After the Julia session exits, traces will be written to `opencl-kernel-profiler.trace` in the current directory, which can be changed using the `CLKP_TRACE_DEST` environment variable. These traces can then be visualized by going to [https://ui.perfetto.dev/](https://ui.perfetto.dev/) and opening the trace file.

`opencl-kernel-profiler` works by intercepting calls to the OpenCL API and recording the execution time of kernels as well as events on the host side. It will also log the OpenCL/SPIR-V source code of the kernels alongside the traced calls to `clEnqueueNDRangekernel`, which can be useful for debugging.
