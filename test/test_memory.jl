facts("OpenCL.Memory") do
    function create_test_buffer()
        ctx = cl.create_some_context()
        queue = cl.CmdQueue(ctx)
        testarray = zeros(Float32, 1000)
        buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
        return (queue, buf, testarray, ctx)
    end

    context("OpenCL.CLMemObject context") do
        _, buf, _, expected = create_test_buffer()

        ctx = cl.context(buf)

        @fact ctx => anything
        @fact isequal(ctx, expected) => true
    end

    context("OpenCL.CLMemObject properties") do
        _, buf, _, _ = create_test_buffer()

        expectations = [
            (:mem_type, cl.CL_MEM_OBJECT_BUFFER),
            (:mem_flags, (:rw, :copy)),
            (:size, sizeof(buf)),
            (:reference_count, 1),
            (:map_count, 0)
        ]

        for expectation in expectations 
            prop, value = expectation
            @fact cl.info(buf, prop) => value prop
        end
    end
end
