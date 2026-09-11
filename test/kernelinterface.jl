import KernelInterface
using OpenCL.OpenCLInterface

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(OpenCLInterface.OpenCLBackend, "OpenCL", OpenCL, CLArray, OpenCL.CLDeviceArray)
