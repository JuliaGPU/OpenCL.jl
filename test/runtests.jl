module TestOpenCL
	using FactCheck
    
    @runtest OpenCL test_platform 
    @runtest OpenCL test_context 
    @runtest OpenCL test_device 
    @runtest OpenCL test_cmdqueue 
    @runtest OpenCL test_event 
    @runtest OpenCL test_buffer 
    @runtest OpenCL test_program 
    @runtest OpenCL test_kernel
	@runtest OpenCL behavior_tests
    exitstatus()

end # module
