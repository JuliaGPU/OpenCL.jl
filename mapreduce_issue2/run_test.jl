#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)

using OpenCL_jll, OpenCL_Headers_jll,
      LLVM_full_jll, Hwloc_jll,
      SPIRV_LLVM_Translator_jll, pocl_jll

# Compile the test program
run(```clang -L $(dirname(OpenCL_jll.libopencl)) -lOpenCL
             -I $(dirname(dirname(OpenCL_Headers_jll.cl_h)))
             test_kernel.c -o test_kernel```)

# Set up library paths
# Note: We DON'T include pocl_jll's lib directory because it has a hardcoded
# rpath to ../share/lib which contains incompatible glibc 2.17 libraries.
# Instead, we use OCL_ICD_FILENAMES to point directly to libpocl.so below.
bindirs = [joinpath(Sys.BINDIR, Base.LIBDIR); joinpath(Sys.BINDIR, Base.PRIVATE_LIBDIR)]
for jll in [OpenCL_jll, SPIRV_LLVM_Translator_jll, Hwloc_jll]
    pushfirst!(bindirs, joinpath(jll.artifact_dir, "lib"))
end

# Determine which SPIR-V file to use
spirv_file = length(ARGS) > 0 ? ARGS[1] : "1_11_opaque.spvasm"

# Assemble SPIR-V if needed
if endswith(spirv_file, ".spvasm")
    spv_out = replace(spirv_file, ".spvasm" => ".spv")
    println("Assembling $spirv_file to $spv_out")
    run(`spirv-as $spirv_file -o $spv_out`)
    spirv_file = spv_out
end

# Run the test
println("Running test with $spirv_file")

run(setenv(`./test_kernel $spirv_file`,
           Dict("LD_LIBRARY_PATH" => join(bindirs, ":"),
                # Preload system libraries to override pocl's bundled glibc 2.17
                "LD_PRELOAD" => "/lib64/libm.so.6 /lib64/libc.so.6",
                #"POCL_DEBUG" => "all",
                #"OCL_ICD_ENABLE_TRACE" => "1",
                "POCL_ARGS_CLANG" => "--ld-path=/usr/bin/ld",
                "POCL_PATH_LD" => "/usr/bin/ld",
                "OCL_ICD_VENDORS" => "",#joinpath(pocl_jll.artifact_dir, "etc", "OpenCL", "vendors"),
                "OCL_ICD_FILENAMES" => pocl_jll.libpocl_path,
                #"LD_DEBUG" => "libs"
                "POCL_PATH_SPIRV_LINK" => ENV["POCL_PATH_SPIRV_LINK"],
                "POCL_PATH_LLVM_SPIRV" => ENV["POCL_PATH_LLVM_SPIRV"],
           )))
