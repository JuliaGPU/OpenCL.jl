skip_tests=Set([
    "sparse",
    "Convert", # Need to opt out of i128
])
KATestSuite.testsuite(OpenCLBackend, "OpenCL", OpenCL, CLArray, CLDeviceArray; skip_tests)
