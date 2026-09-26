import KernelInterface
using OpenCL

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(OpenCLBackend, "OpenCL", OpenCL, CLArray, OpenCL.CLDeviceArray)
