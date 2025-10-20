using Distributed, Test
using OpenCL, pocl_jll
using IOCapture

# KernelAbstractions has a testsuite that isn't part of the main package.
# Include it directly.
const KATestSuite = let
    mod = @eval module $(gensym())
        using ..Test
        import KernelAbstractions
        kernelabstractions = pathof(KernelAbstractions)
        kernelabstractions_root = dirname(dirname(kernelabstractions))
        include(joinpath(kernelabstractions_root, "test", "testsuite.jl"))
    end
    mod.Testsuite
end

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
testf(f, xs...; kwargs...) = GPUArraysTestSuite.compare(f, CLArray, xs...; kwargs...)

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
        if "cl_khr_fp16" in cl.device().extensions && !contains(cl.platform().version, "7.1")
            push!(types, Float16)
            push!(types, ComplexF16)
        end
        return types
    end
end

using Random


## entry point

const targets = []

function runtests(f, name, platform_filter)
    old_print_setting = Test.TESTSET_PRINT_ENABLE[]
    Test.TESTSET_PRINT_ENABLE[] = false

    if isempty(targets)
        for platform in cl.platforms(),
            device in cl.devices(platform)
            if platform_filter !== nothing
                # filter on the name or vendor
                names = lowercase.([platform.name, platform.vendor])
                if !any(contains(platform_filter), names)
                    continue
                end
            end
            push!(targets, (; platform, device))
        end
        if isempty(targets)
            if platform_filter === nothing
                throw(ArgumentError("No OpenCL platforms found"))
            else
                throw(ArgumentError("No OpenCL platforms found matching $platform_filter"))
            end
        end
    end

    try
        # generate a temporary module to execute the tests in
        mod_name = Symbol("Test", rand(1:100), "Main_", replace(name, '/' => '_'))
        mod = @eval(Main, module $mod_name end)
        @eval(mod, using Test, Random, OpenCL)

        let id = myid()
            wait(@spawnat 1 print_testworker_started(name, id))
        end

        # some tests require native execution capabilities
        requires_il = name in ["atomics", "execution", "intrinsics", "kernelabstractions"] ||
                      startswith(name, "gpuarrays/")

        ex = quote
            GC.gc(true)
            Random.seed!(1)
            OpenCL.allowscalar(false)

            @timed @testset $"$name" begin
                @testset "\$(device.name)" for (; platform, device) in $targets
                    cl.platform!(platform)
                    cl.device!(device)

                    if !$requires_il || "cl_khr_il_program" in device.extensions
                        $f()
                    end
                end
            end
        end
        data = Core.eval(mod, ex)
        #data[1] is the testset

        # process results
        cpu_rss = Sys.maxrss()
        compilations = OpenCL.compilations[]
        if VERSION >= v"1.11.0-DEV.1529"
            tc = Test.get_test_counts(data[1])
            passes,fails,error,broken,c_passes,c_fails,c_errors,c_broken =
                tc.passes, tc.fails, tc.errors, tc.broken, tc.cumulative_passes,
                tc.cumulative_fails, tc.cumulative_errors, tc.cumulative_broken
        else
            passes,fails,errors,broken,c_passes,c_fails,c_errors,c_broken =
                Test.get_test_counts(data[1])
        end
        if data[1].anynonpass == false
            data = ((passes+c_passes,broken+c_broken),
                    data[2],
                    data[3],
                    data[4],
                    data[5])
        end
        res = vcat(collect(data), cpu_rss, compilations)

        GC.gc(true)
        res
    finally
        Test.TESTSET_PRINT_ENABLE[] = old_print_setting
    end
end


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


nothing # File is loaded via a remotecall to "include". Ensure it returns "nothing".
