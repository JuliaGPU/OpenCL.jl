using ParallelTestRunner
using Preferences
import OpenCL, pocl_jll
import Test

@info "System information:\n" * sprint(io->OpenCL.versioninfo(io))

## --platform selector
do_platform, platform_filter = ParallelTestRunner.extract_flag!(ARGS, "--platform", nothing)

test_transform = function(test, expr)
    # some tests require native execution capabilities
    requires_il = test in ["atomics", "execution", "intrinsics", "kernelabstractions", "statistics",
                           "linalg", ] ||
                  startswith(test, "gpuarrays/")

    # targets is a global variable that is defined in init_code
    return quote
        if isempty(targets)
            for platform in cl.platforms(),
                device in cl.devices(platform)
                if $(platform_filter) !== nothing
                    # filter on the name or vendor
                    names = lowercase.([platform.name, platform.vendor])
                    if !any(contains($(platform_filter)), names)
                        continue
                    end
                end
                push!(targets, (; platform, device))
            end
            if isempty(targets)
                if $(platform_filter) === nothing
                    throw(ArgumentError("No OpenCL platforms found"))
                else
                    throw(ArgumentError("No OpenCL platforms found matching $($(platform_filter))"))
                end
            end
        end

        @testset "\$(device.name)" for (; platform, device) in targets
            cl.platform!(platform)
            cl.device!(device)

            if !$(requires_il) || "cl_khr_il_program" in device.extensions
                $(expr)
            end
        end
    end
end


# register custom tests that do not correspond to files in the test directory
custom_tests = Dict{String, Expr}()

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
const GPUArraysTestSuite = let
    mod = @eval module $(gensym())
        using ..Test
        import GPUArrays
        gpuarrays = pathof(GPUArrays)
        gpuarrays_root = dirname(dirname(gpuarrays))
        include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
    end
    mod.TestSuite
end

for name in keys(GPUArraysTestSuite.tests)
    test = "gpuarrays/$name"
    custom_tests[test] = test_transform(test, :(GPUArraysTestSuite.tests[$name](CLArray)))
end

function test_filter(test)
    if load_preference(OpenCL, "default_memory_backend") == "svm" &&
       test == "gpuarrays/indexing scalar"
        # GPUArrays' scalar indexing tests assume that indexing is not supported
        return false
    end
    return true
end

const init_code = quote
    using OpenCL, pocl_jll

    OpenCL.allowscalar(false)
    const targets = []

    # GPUArrays has a testsuite that isn't part of the main package.
    # Include it directly.
    const GPUArraysTestSuite = let
        mod = @eval module $(gensym())
            using ..Test
            import GPUArrays
            gpuarrays = pathof(GPUArrays)
            gpuarrays_root = dirname(dirname(gpuarrays))
            include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
        end
        mod.TestSuite
    end

    const device_eltypes = Dict()
    function GPUArraysTestSuite.supported_eltypes(::Type{<:CLArray})
        get!(device_eltypes, cl.device()) do
            types = [Int16, Int32, Int64,
                    Complex{Int16}, Complex{Int32}, Complex{Int64},
                    Float32, ComplexF32]
            if "cl_khr_fp64" in cl.device().extensions
                push!(types, Float64)
                push!(types, ComplexF64)
            end
            if "cl_khr_fp16" in cl.device().extensions
                push!(types, Float16)
                push!(types, ComplexF16)
            end
            return types
        end
    end

    testf(f, xs...; kwargs...) = GPUArraysTestSuite.compare(f, CLArray, xs...; kwargs...)

    ## auxiliary stuff

    # Run some code on-device
    macro on_device(ex...)
        code = ex[end]
        kwargs = ex[1:end-1]

        @gensym kernel
        esc(quote
            let
                function $kernel()
                    $code
                    return
                end

                @opencl $(kwargs...) $kernel()
                cl.finish(cl.queue())
            end
        end)
    end
end


runtests(OpenCL, ARGS; custom_tests, test_filter, init_code, test_transform)
