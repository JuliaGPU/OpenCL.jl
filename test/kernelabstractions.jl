if !in("cl_khr_il_program", cl.device().extensions)
@warn "Skipping KernelAbstractions.jl tests on $(cl.platform().name)"
else

import KernelAbstractions
include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

skip_tests=Set([
    "sparse",
    "Convert", # Need to opt out of i128
])
Testsuite.testsuite(OpenCLBackend, "OpenCL", OpenCL, CLArray, CLDeviceArray; skip_tests)

end
