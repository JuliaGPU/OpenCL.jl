# Device-side exceptions

OpenCL kernels report device-side exceptions to the host as `KernelException`s. The
faulting work-item exits, and the exception is raised at the next synchronization point:
`OpenCL.synchronize()` (or `OpenCL.@sync`), a blocking copy such as `Array(x)`, or
`KernelAbstractions.synchronize`.

```julia-repl
julia> a = OpenCL.zeros(Float32, 1);

julia> @opencl (a -> (a[2] = 1f0; return))(a)

julia> OpenCL.synchronize()
ERROR: KernelException: A BoundsError was thrown by work-item 1×1×1 in work-group 1×1×1 on device cpu-...: Out-of-bounds array access
For more details, run Julia with `-g2`, or launch the kernel with `@opencl debug_level=2`
```

The mailbox is reset before the host exception is thrown, so subsequent kernels can run
normally. Results from a failed kernel may be incomplete. Exception handling does not relax
OpenCL's requirements for convergent work-group barriers.

## Diagnostic detail

Kernels default to Julia's `-g` setting. Use `@opencl debug_level=N`, or the same keyword to
`clfunction`, to select the amount of diagnostic information per kernel:

- `0`: only that an exception occurred, and the work-item's local id and work-group id (both
  1-based).
- `1` (the default): also the type and reason of the exception. These are precise for
  common runtime errors, including bounds errors, domain errors, overflow, and inexact
  conversions; other exceptions are reported by a generic name.
- `2`: also a device-side backtrace.

```julia
@opencl debug_level=2 (a -> (a[2] = 1f0; return))(a)
try
    OpenCL.synchronize()
catch exc
    @show exc.dev exc.name exc.reason
    @show exc.work_item exc.work_group exc.backtrace
end
```

Backtrace entries are `(function, file, line)` tuples. Fields unavailable at the selected
level contain empty strings or empty vectors. The kernel only records which of its throw
sites threw: the compiler keeps a table of these sites with the kernel, with their name,
reason and backtrace, which the host looks up, so a higher level doesn't add code to kernels.

## Synchronization and memory

Each device in an OpenCL context has a host-visible exception mailbox, allocated on first
launch. Queues targeting that device share the mailbox, so checking one queue may wait for
and report a failure from another queue on the same device. `exc.dev` identifies that device.
Only the first faulting work-item records its throw site and position until the host
consumes the report.

Mailbox allocation tries USM host memory, fine-grained SVM, coarse-grained SVM, then a
host-accessible buffer with `cl_ext_buffer_device_address`. The memory must be supported by
the context's devices. Coarse-grained SVM and buffers are mapped only after all queues using
the mailbox have completed, and unmapped before another launch can access them.

If no supported memory mechanism is available, OpenCL.jl warns on first use and runs without
host-side exception reporting. Finalizer-driven synchronization leaves reports pending for
the next ordinary synchronization. Mailboxes and their contexts are currently retained for
the lifetime of the process.
