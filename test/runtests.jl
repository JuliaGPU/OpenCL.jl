using ParallelTestRunner
using Preferences
import OpenCL
import Test

## custom arguments
args = parse_args(ARGS; custom=["platform"])

# `--platform` selects which OpenCL platform to run on, substring-matched against the
# platform name/vendor (e.g. `pocl`, `cuda`, `intel`). The special value `pocl_next`
# loads `pocl_next_jll` — a build of the upcoming PoCL release — instead of the released
# `pocl_jll`, then filters identically to `--platform=pocl` (both register a platform
# with the same name/vendor, differing only in version). We import exactly one PoCL JLL,
# so only a single PoCL platform shows up.
const platform_request = args.custom["platform"] === nothing ? nothing :
                         args.custom["platform"].value
const pocl_pkg = platform_request == "pocl_next" ? :pocl_next_jll : :pocl_jll
const platform_arg = platform_request == "pocl_next" ? Some("pocl") :
                     args.custom["platform"]
@eval import $pocl_pkg

@info "System information:\n" * sprint(io->OpenCL.versioninfo(io))

# determine tests to run
const testsuite = find_tests(@__DIR__)
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
    # targets is a global variable that is defined in init_code
    return quote
        if isempty(targets)
            platform_filter = $(platform_arg)
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

            # Tests run kernels through SPIR-V IL where the device supports it,
            # and otherwise through the spirv2clc OpenCL C source fallback, so
            # they run on any device. Set JULIA_OPENCL_TEST_BACKEND=opencl to
            # force the source path even where IL is available (see init code).
            $(expr)
        end
    end
end
for test in keys(testsuite)
    testsuite[test] = generate_test(test, testsuite[test])
end

const init_worker_code = quote
    using OpenCL, $pocl_pkg

    OpenCL.allowscalar(false)

    # Optionally force how kernels are fed to the driver (:auto/:spirv/:opencl), e.g. to
    # exercise the spirv2clc OpenCL C source path on a device that also supports IL programs.
    let b = get(ENV, "JULIA_OPENCL_TEST_BACKEND", "")
        isempty(b) || OpenCL.program_backend!(Symbol(b))
    end

    const targets = []

    # GPUArrays has a testsuite that isn't part of the main package.
    # Include it directly.
    const GPUArraysTestSuite = let
        mod = @eval module $(gensym())
            using Test
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

const init_code = quote
    using OpenCL, $pocl_pkg

    # bring used symbols into the temporary module
    import ..GPUArraysTestSuite, ..testf
    import ..@on_device, ..targets
end

# avoid handle exhaustion on Windows by running each test in a separate process (pocl/pocl#1941)
function test_worker(_, init_worker_code)
    if Sys.iswindows()
        addworker(; init_worker_code)
    else
        nothing
    end
end

runtests(OpenCL, args; testsuite, init_code, init_worker_code, test_worker)
