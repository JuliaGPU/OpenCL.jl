# OpenCL.jl

OpenCL bindings for Julia

**Build status**: [![][buildbot-julia05-img]][buildbot-julia05-url] [![][buildbot-julia06-img]][buildbot-julia06-url] [![][buildbot-travis-img]][buildbot-travis-url]

**Code coverage**: [![](https://img.shields.io/coveralls/JuliaGPU/OpenCL.jl.png)](https://coveralls.io/r/JuliaGPU/OpenCL.jl?branch=master)

[buildbot-julia05-img]: http://ci.maleadt.net/shields/build.php?builder=OpenCL-julia05-x86-64bit&name=julia%200.5
[buildbot-julia05-url]: http://ci.maleadt.net/shields/url.php?builder=OpenCL-julia05-x86-64bit
[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=OpenCL-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=OpenCL-julia06-x86-64bit
[buildbot-travis-img]: https://travis-ci.org/JuliaGPU/OpenCL.jl.svg?branch=master
[buildbot-travis-url]: https://travis-ci.org/JuliaGPU/OpenCL.jl?branch=master

Julia interface for the OpenCL parallel computation API

This package aims to be a complete solution for OpenCL programming in Julia, similar in scope to [PyOpenCL] for Python.
It provides a high level api for OpenCL to make programing GPU's and multicore CPU's much less onerous.

OpenCL.jl provides access to OpenCL API versions 1.0, 1.1, 1.2 and 2.0.

#### This package is based off the work of others:
  * [PyOpenCL] by Andreas Klockner
  * [oclpb]    by Sean Ross
  * [Boost.Compute] by Kyle Lutz
  * [rust-opencl]

[PyOpenCL]: http://mathema.tician.de/software/pyopencl/
[oclpb]: https://github.com/srossross/oclpb
[Boost.Compute]:https://github.com/kylelutz/compute
[rust-opencl]: https://github.com/luqmana/rust-opencl

OpenCL.jl has had contributions from [many developers](https://github.com/JuliaGPU/OpenCL.jl/graphs/contributors).

## Currently supported Julia versions
- Julia `v"0.4.x"` is supported on the `release-0.4` branch and the OpenCL.jl versions `v"0.4.x"`. Only bug-fixes will be applied.
- Julia `v"0.5.x"` is supported on the `master` branch and the OpenCL.jl versions `v"0.5.x"`.
- Julia `v"0.6.x"` is *experimentally* supported on the `master` branch and the OpenCL.jl versions `v"0.5.x"`.

### Discontinued support
- Julia `v"0.3.x"` was supported on OpenCL.jl versions `v"0.3.x"`. It should still be installable and work.

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

a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=a)
b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=b)
c_buff = cl.Buffer(Float32, ctx, :w, length(a))

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

## Translation


Here's a rough translation between the OpenCL API in C to this Julia version. Optional arguments are indicated by `[name?]` (see `clCreateBuffer`, for example). For a quick reference to the C version, see [the Khronos quick reference card](https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf) or the [online specification](https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html).

Whenever there's an `cl.info(object, :info_name)`, you can also call `info_name(object)`. This will give the same information, but is documented and type stable.


### Platform and Devices


| C                   | Julia                                                                 | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|---------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `clGetPlatformIDs`  | `cl.platforms()`                                                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `clGetPlatformInfo` | `cl.info(platform, :symbol)`                                          | Platform info: `:profile`, `:version`, `:name`, `:vendor`, `:extensions`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `clGetDeviceIDs`    | `cl.devices()`, `cl.devices(platform)`, `cl.devices(:type)`           | Device types: `:all`, `:cpu`, `:gpu`, `:accelerator`, `:custom`, `:default`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `clGetDeviceInfo`   | `cl.info(device, :symbol)`                                            | Device info: `:driver_version`, `:version`, `:profile`, `:extensions`, `:platform`, `:name`, `:device_type`, `:has_image_support`, `:queue_properties`, `:has_queue_out_of_order_exec`, `:has_queue_profiling`, `:has_native_kernel`, `:vendor_id`, `:max_compute_units`, `:max_work_item_size`, `:max_clock_frequency`, `:address_bits`, `:max_read_image_args`, `:max_write_image_args`, `:global_mem_size`, `:max_mem_alloc_size`, `:max_const_buffer_size`, `:local_mem_size`, `:has_local_mem`, `:host_unified_memory`, `:available`, `:compiler_available`, `:max_work_group_size`, `:max_work_item_dims`, `:max_parameter_size`, `:profiling_timer_resolution`,  `:max_image2d_shape`, `:max_image3d_shape` |
| `clCreateContext`   | `cl.context(queue)`, `cl.context(CLMemObject)`, `cl.context(CLArray)` |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `clReleaeContext`   | `cl.release!`                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |


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


| C                              | Julia                                                                                                                                                    | Notes                                                                            |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `clCreateKernel`               | `cl.Kernel(program, "kernel_name")`                                                                                                                      |                                                                                  |
| `clGetKernelInfo`              | `cl.info(kernel, :symbol)`                                                                                                                               | Kernel info: `:name`, `:num_args`, `:reference_count`, `:program`, `:attributes` |
| `clEnqueueNDRangeKernel`       | `cl.enqueue_kernel(queue, kernel, global_work_size)`, `cl.enqueue_kernel(queue, kernel, global_work_size, local_work_size; global_work_offset, wait_on)` |                                                                                  |
| `clSetKernelArg`               | `cl.set_arg!(kernel, idx, arg)`                                                                                                                          | `idx` starts at 1                                                                |
| `clCreateUserEvent`            | `cl.UserEvent(ctx; retain)`                                                                                                                              |                                                                                  |
| `clGetEventInfo`               | `cl.info(event, :symbol)`                                                                                                                                | Event info: `:context`, `:command_queue`, `:reference_count`, `:command_type`, `:status`, `:profile_start`, `:profile_end`, `:profile_queued`, `:profile_submit`, `:profile_duration`
| `clWaitForEvents`              | `cl.wait(event)`, `cl.wait(events)`                                                                                                                      |
| `clEnqueueMarkerWithWaitList`  | `cl.enqueue_marker_with_wait_list(queue, wait_for)`                                                                                                      | |
| `clEnqueueBarrierWithWaitList` | `cl.enqueue_barrier_with_wait_list(queue, wait_for)`                                                                                                     | |
