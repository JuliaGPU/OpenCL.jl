function create_test_buffer()
    testarray = zeros(Float32, 1000)
    cl.Buffer(Float32, cl.context(), length(testarray), (:rw, :copy), hostbuf=testarray)
end

@testset "Memory" begin
    @testset "context" begin
        buf = create_test_buffer()

        ctx = cl.context(buf)

        @test ctx != nothing
        @test isequal(ctx, cl.context()) != nothing
    end

    @testset "properties" begin
        buf = create_test_buffer()

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
