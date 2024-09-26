if !in("cl_khr_il_program", cl.device().extensions)
@warn "Skipping KernelAbstractions.jl tests on $(cl.platform().name)"
else

skip_tests=Set([
    "sparse",
    "Convert", # Need to opt out of i128
])
KATestSuite.testsuite(OpenCLBackend, "OpenCL", OpenCL, CLArray, CLDeviceArray; skip_tests)

end
