module TestOpenCL
    using FactCheck

    const tests = map(ARGS) do test
        test_file = "test_$test.jl"
        if !isfile(test_file)
            warn("Could not find $test_file")
            return ""
        end
        return test_file
    end

    macro test_opencl()
        for test in tests
            if test != ""
                include(test)
            end
        end
    end
    
    @test_opencl
    exitstatus()
end # module
