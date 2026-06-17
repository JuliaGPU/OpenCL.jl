# SPIRVIntrinsics.jl

*Reusable intrinsic functions for SPIRV-based programming packages in Julia.*

This package provides Julia functions that compile down to compiler intrinsics
usable in SPIR-V code. It is used by packages such as OpenCL.jl and oneAPI.jl.


## Usage

End users can use this package by calling the intrinsics from Julia kernel code.
For documentation, refer to the user-facing package (OpenCL.jl, oneAPI.jl).

For developers wanting to integrate this package, there are a couple of
considerations:

- to prevent accidental use of the intrinsics in non-SPIR-V code, method overlay
  tables are used, implying that `SPIRVIntrinsics.method_table` should be used
  for compilation;

- exported functionality is meant to be re-exported, while non-exported
  functionality is typically meant to be accessed via the module of the
  user-facing package. This can be accomplished using some metaprogramming:

  ```julia
  # re-export functionality from SPIRVIntrinsics
  for name in names(SPIRVIntrinsics)
      name == :SPIRVIntrinsics && continue
      @eval export $name
  end

  # import all the others so that the user can refer to them through the OpenCL module
  for name in names(SPIRVIntrinsics; all=true)
      # bring all the names of this module in scope
      name in (:SPIRVIntrinsics, :eval, :include) && continue
      startswith(string(name), "#") && continue
      @eval begin
          using .SPIRVIntrinsics: $name
      end
  end
  ```


## SPIR-V representation

Intrinsics that map to core SPIR-V operations should be encoded using LLVM's
SPIR-V wrapper builtins, such as `__spirv_AtomicIAdd` or
`__spirv_GroupNonUniformShuffle`. This keeps the emitted LLVM IR independent of
OpenCL C builtin spellings.

Some math and integer functions intentionally keep OpenCL.std names. In SPIR-V,
those operations are represented through the OpenCL extended instruction set,
so the OpenCL spelling is the SPIR-V-level contract rather than an OpenCL.jl API
dependency.
