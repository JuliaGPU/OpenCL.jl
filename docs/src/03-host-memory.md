# Sharing memory with the host

`CLArray`s normally live in memory managed by OpenCL, and moving data between them and
Julia `Array`s requires a copy (`CLArray(a)`, `Array(b)`, `copyto!`). Depending on the
device and on the kind of memory, it is also possible to share memory between both kinds
of arrays without copying, using `unsafe_wrap`.

## Using host memory on the device

`unsafe_wrap(CLArray, a)` wraps a `CLArray` around the memory of an existing `Array`.
Operations on the `CLArray` execute on the device, but operate directly on the memory of
the original array:

```julia-repl
julia> using OpenCL, pocl_jll

julia> a = Float32[1, 2, 3, 4];

julia> b = unsafe_wrap(CLArray, a)
4-element CLArray{Float32, 1, OpenCL.cl.SharedVirtualMemory}:
 1.0
 2.0
 3.0
 4.0

julia> b .= sqrt.(b);

julia> cl.finish(cl.queue())

julia> a
4-element Vector{Float32}:
 1.0
 1.4142135
 1.7320508
 2.0
```

This makes it possible to use GPU-style array code on ordinary arrays. For example, with
PoCL's CPU device, the broadcast above runs in parallel over all CPU cores, without
copying `a` into a separate device buffer first.

Wrapping host memory requires a device that can directly access memory allocated by the
host operating system, as indicated by support for *fine-grained system shared virtual
memory* (e.g., PoCL's CPU device, or Intel's CPU runtime) or for *shared system
allocations* in the `cl_intel_unified_shared_memory` extension. On other devices,
`unsafe_wrap` throws an `ArgumentError`, and the data has to be copied using `CLArray(a)`
instead.

It is also possible to wrap a raw pointer, `unsafe_wrap(CLArray, ptr, dims)`, in which case
the caller is responsible for keeping the memory alive for as long as the `CLArray` is used.
When wrapping an `Array`, the `CLArray` keeps it alive.

Atomic operations on 8- or 16-bit elements (e.g., `Int8` or `Float16`) are implemented with
32-bit atomics on the aligned 4-byte word that contains the element. When using them on
wrapped memory, the caller has to make sure that these words are accessible in their
entirety, also where they extend past the end of the wrapped memory. Memory allocated by
OpenCL.jl is padded to a multiple of 4 bytes to guarantee this.

## Using device memory on the host

The opposite direction, `unsafe_wrap(Array, b)`, wraps an `Array` around the memory of a
`CLArray`. This is possible for arrays that are backed by memory the host can access:
unified shared or host memory (`cl.UnifiedSharedMemory`, `cl.UnifiedHostMemory`), shared
virtual memory (`cl.SharedVirtualMemory`), or host memory that was wrapped as shown above.
The memory type can be selected when allocating the array, e.g.,
`CLArray{Float32, 1, cl.UnifiedSharedMemory}(undef, 4)`.

!!! warning

    The returned `Array` does **not** keep the `CLArray` alive. The caller has to keep a
    reference to the `CLArray` for as long as the `Array`, or anything derived from it, is
    used; otherwise the `Array` may end up referring to freed memory.

## Caveats

These functions are unsafe, because the resulting arrays alias memory that is managed
elsewhere:

- Device operations execute asynchronously. Wait for them to finish, e.g., using
  `cl.finish(cl.queue())`, before accessing the memory on the host.
- Wrapped memory must not be freed or reallocated while it is in use. Do not `resize!` the
  original array while it is wrapped; resizing the wrapper itself is not supported.
- Wrapping the same memory multiple times creates independent arrays, whose operations are
  not synchronized with each other.
- Coarse-grained shared virtual memory is only accessible from the host while it is
  mapped. After using the `CLArray` on the device, an `Array` obtained from it must not be
  accessed until an operation on the `CLArray` has mapped it again (e.g., indexing it, or
  calling `unsafe_wrap(Array, b)` again). This does not apply to other kinds of memory.
