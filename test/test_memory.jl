function create_test_buffer()
    ctx = cl.Context(device)
    queue = cl.CmdQueue(ctx)
    testarray = zeros(Float32, 1000)
    buf = cl.Buffer(Float32, ctx, length(testarray), (:rw, :copy), hostbuf=testarray)
    return (queue, buf, testarray, ctx)
end

@testset "OpenCL.Memory" begin
    @testset "OpenCL.CLMemObject context" begin
        _, buf, _, expected = create_test_buffer()

        ctx = cl.context(buf)

        @test ctx != nothing
        @test isequal(ctx, expected) != nothing
    end

    @testset "OpenCL.CLMemObject properties" begin
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
            @test cl.info(buf, prop) == value
        end
    end
end
