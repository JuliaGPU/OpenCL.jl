### >> THIS PACKAGE NEEDS A MAINTAINER << ###

Please reach out to @juliohm if you would like to be the new maintainer.

# OpenCL.jl

[![][buildkite-img]][buildkite-url]

[buildkite-img]: https://badge.buildkite.com/6b2a46bff67692115dea3ad5a275d2f80777a5a99ffe42adb0.svg
[buildkite-url]: https://buildkite.com/julialang/opencl-dot-jl

*Julia interface for the OpenCL parallel computation API*

This package aims to be a complete solution for OpenCL programming in Julia, similar in scope to [PyOpenCL] for Python. It provides a high level API for OpenCL to make programing hardware accelerators, such as GPUs, FPGAs, and DSPs, as well as multicore CPUs much less onerous.

OpenCL.jl provides access to [OpenCL API](https://www.khronos.org/registry/OpenCL/) versions 1.0, 1.1, 1.2 and 2.0.

## Installation
1. Install an OpenCL driver. (If you're on macOS, OpenCL is either already available or unsupported.)
2. Add OpenCL to your Julia environment:

```julia
using Pkg
Pkg.add("OpenCL")
```

## Basic example: vector add

**Note:** We use `cl.create_compute_context()` here which only considers GPUs and CPUs.

```julia
using LinearAlgebra
using OpenCL

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

a_buff = cl.Buffer(Float32, ctx, length(a), (:r, :copy), hostbuf=a)
b_buff = cl.Buffer(Float32, ctx, length(b), (:r, :copy), hostbuf=b)
c_buff = cl.Buffer(Float32, ctx, length(a), :w)

p = cl.Program(ctx, source=sum_kernel) |> cl.build!
k = cl.Kernel(p, "sum")

queue(k, size(a), nothing, a_buff, b_buff, c_buff)

r = cl.read(queue, c_buff)

if isapprox(norm(r - (a+b)), zero(Float32))
    @info "Success!"
else
    @error "Norm should be 0.0f"
end
```

## More examples
You may want to check out the `examples` folder. Either `git clone` the repository to your local machine or navigate to the OpenCL.jl install directory via
```julia
using OpenCL
cd(joinpath(dirname(pathof(OpenCL)), ".."))
```

Otherwise, feel free to take a look at the Jupyter notebooks below
  * [OpenCL Fractals]
  * [GPU Buffer Transpose]
  * [Low Level API]

[OpenCL Fractals]:http://nbviewer.ipython.org/7517923
[GPU Buffer Transpose]:http://nbviewer.ipython.org/7517952
[Low Level API]:http://nbviewer.ipython.org/7452048

## Credit

This package is heavily influenced by the work of others:

  * [PyOpenCL] by Andreas Klockner
  * [oclpb]    by Sean Ross
  * [Boost.Compute] by Kyle Lutz
  * [rust-opencl]

[PyOpenCL]: http://mathema.tician.de/software/pyopencl/
[oclpb]: https://github.com/srossross/oclpb
[Boost.Compute]:https://github.com/kylelutz/compute
[rust-opencl]: https://github.com/luqmana/rust-opencl

## Documentation: API


Here's a rough translation between the OpenCL API in C to this Julia version. Optional arguments are indicated by `[name?]` (see `clCreateBuffer`, for example). For a quick reference to the C version, see [the Khronos quick reference card](https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf).


### Platform and Devices


| C                   | Julia                                                       | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|---------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `clGetPlatformIDs`  | `cl.platforms()`                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `clGetPlatformInfo` | `cl.info(platform, :symbol)`                                | Platform info: `:profile`, `:version`, `:name`, `:vendor`, `:extensions`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `clGetDeviceIDs`    | `cl.devices()`, `cl.devices(platform)`, `cl.devices(:type)` | Device types: `:all`, `:cpu`, `:gpu`, `:accelerator`, `:custom`, `:default`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `clGetDeviceInfo`   | `cl.info(device, :symbol)`                                  | Device info: `:driver_version`, `:version`, `:profile`, `:extensions`, `:platform`, `:name`, `:device_type`, `:has_image_support`, `:queue_properties`, `:has_queue_out_of_order_exec`, `:has_queue_profiling`, `:has_native_kernel`, `:vendor_id`, `:max_compute_units`, `:max_work_item_size`, `:max_clock_frequency`, `:address_bits`, `:max_read_image_args`, `:max_write_image_args`, `:global_mem_size`, `:max_mem_alloc_size`, `:max_const_buffer_size`, `:local_mem_size`, `:has_local_mem`, `:host_unified_memory`, `:available`, `:compiler_available`, `:max_work_group_size`, `:max_work_item_dims`, `:max_parameter_size`, `:profiling_timer_resolution`,  `:max_image2d_shape`, `:max_image3d_shape` |
| `clCreateContext`   | `cl.context(queue)`, `cl.context(CLMemObject), `cl.context(CLArray)` | |
| `clReleaeContext`   | `cl.release!` | |


### Buffers


| C                      | Julia                                                                                                       | Notes                                                      |
|------------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| `clCreateBuffer`       | `cl.Buffer(type, context, [length?]; [hostbuf?])`, `cl.Buffer(type, context, flags, [length?]; [hostbuf?])` | Memory flags: `:rw`, `:r`, `:w`, `:use`, `:alloc`, `:copy` |
| `clEnqueueCopyBuffer`  | `cl.copy!(queue, destination, source)`                                                                      |                                                            |
| `clEnqueueFillBuffer`  | `cl.enqueue_fill_buffer(queue, buffer, pattern, offset, nbytesm wait_for)`                                  |                                                            |
| `clEnqueueReadBuffer`  | `cl.enqueue_read_buffer(queue, buffer, hostbuf, dev_offset, wait_for, is_blocking)`                         |                                                            |
| `clEnqueueWriteBuffer` | `cl.enqueue_write_buffer(queue, buffer, hostbuf, byte_count, offset, wait_for, is_blocking)`                |                                                            |


### Program Objects


| C                             | Julia                        | Notes                                                                                                                           |
|-------------------------------|------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `clCreateProgramWithSource`   | `cl.Program(ctx; source)`    |                                                                                                                                 |
| `clCreateProgramWithBinaries` | `cl.Program(ctx; binaries)`  |                                                                                                                                 |
| `clReleaseProgram`            | `cl.release!`                |                                                                                                                                 |
| `clBuildProgram`              | `cl.build!(progrm, options)` |                                                                                                                                 |
| `clGetProgramInfo`            | `cl.info(program, :symbol)`  | Program info: `:reference_count`, `:devices`, `:context`, `:num_devices`, `:source`, `:binaries`, `:build_log`, `:build_status` |


### Kernel and Event Objects


| C | Julia | Notes |
| --- | --- | ----- |
| `clCreateKernel` | `cl.Kernel(program, "kernel_name")` | |
| `clGetKernelInfo` | `cl.info(kernel, :symbol)` | Kernel info: `:name`, `:num_args`, `:reference_count`, `:program`, `:attributes` |
| `clEnqueueNDRangeKernel` | `cl.enqueue_kernel(queue, kernel, global_work_size)`, `cl.enqueue_kernel(queue, kernel, global_work_size, local_work_size; global_work_offset, wait_on)` | |
| `clSetKernelArg` | `cl.set_arg!(kernel, idx, arg)` | `idx` starts at 1 |
| `clCreateUserEvent` | `cl.UserEvent(ctx; retain)`  | |
| `clGetEventInfo`    | `cl.info(event, :symbol)`    | Event info: `:context`, `:command_queue`, `:reference_count`, `:command_type`, `:status`, `:profile_start`, `:profile_end`, `:profile_queued`, `:profile_submit`, `:profile_duration`
| `clWaitForEvents`   | `cl.wait(event)`, `cl.wait(events)` |
| `clEnqueueMarkerWithWaitList` | `cl.enqueue_marker_with_wait_list(queue, wait_for)` | |
| `clEnqueueBarrierWithWaitList` | `cl.enqueue_barrier_with_wait_list(queue, wait_for)` | |
