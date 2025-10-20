using ParallelTestRunner
using Preferences
import OpenCL, pocl_jll
import Test

@info "System information:\n" * sprint(io->OpenCL.versioninfo(io))

## custom arguments
args = parse_args(ARGS; custom=["platform"])

# determine tests to run
const testsuite = find_tests(pwd())
## GPUArrays test suite: not part of the main package
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
    testsuite[test] = :(GPUArraysTestSuite.tests[$name](CLArray))
end
## filter
if filter_tests!(testsuite, args)
    if load_preference(OpenCL, "default_memory_backend") == "svm"
        # GPUArrays' scalar indexing tests assume that indexing is not supported
        delete!(testsuite, "gpuarrays/indexing scalar")
        return false
    end
end

# wrap tests in device loops
function generate_test(test, expr)
    # some tests require native execution capabilities
    requires_il = test in ["atomics", "execution", "intrinsics", "kernelabstractions",
                           "statistics", "linalg", ] ||
                  startswith(test, "gpuarrays/")

    # targets is a global variable that is defined in init_code
    return quote
        if isempty(targets)
            platform_filter = $(args.custom["platform"])
            for platform in cl.platforms(),
                device in cl.devices(platform)
                if platform_filter !== nothing
                    # filter on the name or vendor
                    names = lowercase.([platform.name, platform.vendor])
                    if !any(contains(platform_filter.value), names)
                        continue
                    end
                end
                push!(targets, (; platform, device))
            end
            if isempty(targets)
                if platform_filter !== nothing
                    throw(ArgumentError("No OpenCL platforms found"))
                else
                    throw(ArgumentError("No OpenCL platforms found matching $(platform_filter.value)"))
                end
            end
        end

        @testset "$(device.name)" for (; platform, device) in targets
            cl.platform!(platform)
            cl.device!(device)

            if !$(requires_il) || "cl_khr_il_program" in device.extensions
                $(expr)
            end
        end
    end
end
for test in keys(testsuite)
    testsuite[test] = generate_test(test, testsuite[test])
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


runtests(OpenCL, args; testsuite, init_code)
