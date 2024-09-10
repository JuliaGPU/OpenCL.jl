# OpenCL.jl release notes


## In development: OpenCL.jl v1.0

This version is a major rewrite of the package, with a focus on unifying the package with
the rest of the JuliaGPU ecosystem.

Breaking changes:

- Context, device and queue arguments have been removed from most APIs, and are now stored
  in task-local storage. These values can be queried (`cl.platform()`, `cl.device()`, etc)
  and set (`cl.platform!(platform)`, `cl.device!(device)`, etc) as needed.
- As part of the above change, questionable APIs like `cl.create_some_context()` and
  `cl.devices()` have been removed.
- The `Buffer` constructor has switched the `length` and `flags` around, for ambiguity
  reasons. The `length` argument is now also mandatory.
- The `cl.info` method, and the `getindex` overloading to access properties of OpenCL
  objects, have been replaced by `getproperty` overloading on the objects themselves
  (e.g., `cl.info(dev, :name)` and `dev[:name]` are now simply `dev.name`).
- The blocking `cl.launch` has been replaced by a nonblocking `cl.call`, while also removing
  the `getindex`-overloading shorthand.
- Argument conversion has been removed; the user should make sure Julia arguments passed to
  kernels match the OpenCL argument types (i.e., no empty types, 4-element tuples for
  a 3-element `float3` arguments).


New features:

- Loading SPIR-V IL programs is now supported: `cl.Program(il=...)`.
- Support for POCL has been added. Together with a rework of `OpenCL_jll`, this makes it
  possible to do `using OpenCL, pocl_jll` and have the POCL platform available.
