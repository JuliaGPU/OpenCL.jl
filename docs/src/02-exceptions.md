# Device-side exceptions

OpenCL kernels report device-side exceptions to the host as `KernelException`s. The
faulting work-item exits, and the exception is raised at the next synchronization point:
`cl.finish`, a blocking copy such as `Array(x)`, or `KernelAbstractions.synchronize`.

```julia-repl
julia> a = OpenCL.zeros(Float32, 1);

julia> @opencl (a -> (a[2] = 1f0; return))(a)

julia> cl.finish(cl.queue())
ERROR: KernelException: A BoundsError was thrown on device cpu-...: Out-of-bounds array access
For more details, run Julia with `-g2`, or launch the kernel with `@opencl debug_level=2`
```

The mailbox is reset before the host exception is thrown, so subsequent kernels can run
normally. Results from a failed kernel may be incomplete. Exception handling does not relax
OpenCL's requirements for convergent work-group barriers.

## Diagnostic detail

Kernels default to Julia's `-g` setting. Use `@opencl debug_level=N`, or the same keyword to
`clfunction`, to select the amount of diagnostic information per kernel:

- `0`: records only that an exception occurred.
- `1` (the default): also records the type and reason for common runtime errors, including
  bounds errors, domain errors, overflow, and inexact conversions.
- `2`: also records the work-item's local id and work-group id (both 1-based), a name for
  other exceptions, and a device-side backtrace.

```julia
@opencl debug_level=2 (a -> (a[2] = 1f0; return))(a)
try
    cl.finish(cl.queue())
catch exc
    @show exc.dev exc.name exc.reason
    @show exc.work_item exc.work_group exc.backtrace
end
```

Backtrace entries are `(function, file, line)` tuples. Fields unavailable at the selected
level contain empty strings, empty vectors, or zero coordinates. Names, reasons, function
names, and file paths are limited to 63, 191, 127, and 127 bytes respectively; backtraces
contain at most 16 frames. Level 2 adds more code to throwing paths and is intended for
debugging.

## Synchronization and memory

Each device in an OpenCL context has a host-visible exception mailbox, allocated on first
launch. Queues targeting that device share the mailbox, so checking one queue may wait for
and report a failure from another queue on the same device. `exc.dev` identifies that device.
Only one faulting work-item records detailed diagnostics until the host consumes the report.
Its launch identifier and coordinates distinguish it from other work-items and later kernels.

Mailbox allocation tries USM host memory, fine-grained SVM, coarse-grained SVM, then a
host-accessible buffer with `cl_ext_buffer_device_address`. The memory must be supported by
the context's devices. Coarse-grained SVM and buffers are mapped only after all queues using
the mailbox have completed, and unmapped before another launch can access them.

If no supported memory mechanism is available, OpenCL.jl warns on first use and runs without
host-side exception reporting. Finalizer-driven synchronization leaves reports pending for
the next ordinary synchronization. Mailboxes and their contexts are currently retained for
the lifetime of the process.
