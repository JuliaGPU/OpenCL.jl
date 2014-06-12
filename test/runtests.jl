module TestOpenCL
	using FactCheck

	@runtest OpenCL test_platform test_context test_device test_cmdqueue test_event test_buffer test_program test_kernel

	@runtest OpenCL behavior_tests
    exitstatus()

end # module