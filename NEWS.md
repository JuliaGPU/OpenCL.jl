# OpenCL.jl release notes


## OpenCL.jl v1.0

This version is a major rewrite of the package, with a focus on unifying the package with
the rest of the JuliaGPU ecosystem.

### `CLArray` and `clcall`

One major change is that memory management should now be done using `CLArray` objects, which
are automatically converted to pointers when invoking kernels using the new `clcall`
function:

```julia-repl
julia> using OpenCL, pocl_jll

julia> const source = """
          __kernel void memset(__global float *arr, float val, int len) {
             int i = get_global_id(0);
             if (i < len) {
                arr[i] = val;
             }
           }""";

julia> arr = CLArray{Float32}(undef, 4)
4-element CLArray{Float32, 1}:
 0.0
 0.0
 0.0
 0.0

julia> prog = cl.Program(; source) |> cl.build!;

julia> kern = cl.Kernel(prog, "memset");

julia> clcall(kern, Tuple{Ptr{Float32}, Float32, Int32},
              arr, 42.0, length(arr); global_size=(length(arr),));

julia> arr
4-element CLArray{Float32, 1}:
 42.0
 42.0
 42.0
 42.0
```

### Julia kernel compilation

On devices that support SPIR-V IL, Julia kernels can be automatically compiled and
executed using the `@opencl` macro:

```julia-repl
julia> using OpenCL, pocl_jll

julia> function memset(arr, val)
           i = get_global_id(1)
           if i <= length(arr)
              arr[i] = val
           end
           return
       end;

julia> arr = CLArray{Float32}(undef, 4)
4-element CLArray{Float32, 1}:
 0.0
 0.0
 0.0
 0.0

julia> @opencl global_size=length(arr) memset(arr, 42.0);

julia> arr
4-element CLArray{Float32, 1}:
 42.0
 42.0
 42.0
 42.0
```

Not many back-ends support SPIR-V IL, so the pocl_jll is provided for users to play with
this functionality. Currently, our build of pocl only supports CPUs, but that may be
extended to GPUs in the future. At the same time, it may become possible to cross-compile
the generated SPIR-V IL back to OpenCL C code for execution on other devices.

### Low-level changes

To make the above possible, many changes have been made to the low-level APIs:

- Context, device and queue arguments have been removed from most APIs, and are now stored
  in task-local storage. These values can be queried (`cl.platform()`, `cl.device()`, etc)
  and set (`cl.platform!(platform)`, `cl.device!(device)`, etc) as needed.
- As part of the above change, questionable APIs like `cl.create_some_context()` and
  `cl.devices()` have been removed.
- The `Buffer` API has been completely reworked. It now only provides low-level
  functionality, such as `unsafe_copyto!` or `unsafe_map!`, while high-level functionality
  like `copy!` is implemented for the `CLArray` type.
- The `cl.info` method, and the `getindex` overloading to access properties of OpenCL
  objects, have been replaced by `getproperty` overloading on the objects themselves
  (e.g., `cl.info(dev, :name)` and `dev[:name]` are now simply `dev.name`).
- The blocking `cl.launch` has been replaced by a nonblocking `cl.call`, while also removing
  the `getindex`-overloading shorthand. However, it's recommended to use the newly-added
  `cl.clcall` function, which takes an additional tuple type argument and performs automatic
  conversions of arguments to those types. This makes it possible to pass a `CLArray` to an
  OpenCL C function expecting Buffer-backed pointers, for example.
- Argument conversion has been removed; the user should make sure Julia arguments passed to
  kernels match the OpenCL argument types (i.e., no empty types, 4-element tuples for
  a 3-element `float3` arguments).
- The `to_host` function has been replaced by simply calling `Array` on the `CLArray`.
- Queue and execution capabilities of a device are now to be queried using dedicated
  functions, `cl.queue_properties` and `cl.exec_capabilities`.

### Future work

More breaking changes are expected in the future, and this release is mostly intended to
make it possible to experiment with the new features, in order to determine whether
OpenCL.jl could be used to power the next generation of KernelAbstraction.jl's CPU back-end.
