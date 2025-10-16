using ParallelTestRunner
using Preferences
import OpenCL, pocl_jll
import Test

@info "System information:\n" * sprint(io->OpenCL.versioninfo(io))

## --platform selector
do_platform, platform_filter = ParallelTestRunner.extract_flag!(ARGS, "--platform", nothing)

custom_record_init = quote
    import ParallelTestRunner: Test
    struct OpenCLTestRecord <: ParallelTestRunner.AbstractTestRecord
        # TODO: Would it be better to wrap "ParallelTestRunner.TestRecord "
        value::Any          # AbstractTestSet or TestSetException
        output::String      # captured stdout/stderr

        # stats
        time::Float64
        bytes::UInt64
        gctime::Float64
        rss::UInt64
    end
    function ParallelTestRunner.memory_usage(rec::OpenCLTestRecord)
        return rec.rss
    end
    function ParallelTestRunner.test_IOContext(::Type{OpenCLTestRecord}, stdout::IO, stderr::IO, lock::ReentrantLock, name_align::Int64)
        return ParallelTestRunner.test_IOContext(ParallelTestRunner.TestRecord, stdout, stderr, lock, name_align)
    end

    const targets = []
    using OpenCL, IOCapture

    function ParallelTestRunner.execute(::Type{OpenCLTestRecord}, mod, f, name, color, (; platform_filter))
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

        # some tests require native execution capabilities
        requires_il = name in ["atomics", "execution", "intrinsics", "kernelabstractions"] ||
                      startswith(name, "gpuarrays/")

        data = @eval mod begin
            GC.gc(true)
            Random.seed!(1)
            OpenCL.allowscalar(false)

            mktemp() do path, io
                stats = redirect_stdio(stdout=io, stderr=io) do
                    @timed try
                        @testset $(Expr(:$, :name)) begin
                            @testset "\$(device.name)" for (; platform, device) in $(Expr(:$, :targets))
                                cl.platform!(platform)
                                cl.device!(device)

                                if !$(Expr(:$, :requires_il)) || "cl_khr_il_program" in device.extensions
                                    $(Expr(:$, :f))
                                end
                            end
                        end
                    catch err
                        isa(err, Test.TestSetException) || rethrow()

                        # return the error to package it into a TestRecord
                        err
                    end
                end
                close(io)
                output = read(path, String)
                (; testset=stats.value, output, stats.time, stats.bytes, stats.gctime)

            end
        end

        # process results
        rss = Sys.maxrss()
        record = OpenCLTestRecord(data..., rss)

        GC.gc(true)
        return record
    end
end # quote
eval(custom_record_init)

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
    custom_tests["GPUArraysTestSuite/$name"] = :(GPUArraysTestSuite.tests[$name](CLArray))
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

runtests(OpenCL, ARGS; custom_tests, test_filter, init_code, custom_record_init,
                       RecordType=OpenCLTestRecord, custom_args=(;platform_filter))
