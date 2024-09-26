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


## OpenCL intrinsics

The current set of intrinsics implemented by this package are OpenCL intrinsics,
assuming that the generated LLVM IR will be compiled to SPIR-V using the
Khronos LLVM to SPIR-V translator. That tool will take care of the conversion to
actual SPIR-V intrinsics.
