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

platform_filter = platform_arg
platform_selected(p) = platform_filter === nothing ||
    any(contains(platform_filter.value), lowercase.([p.name, p.vendor]))

ispocl(p) = occursin("portable", lowercase(p.name))

# short, stable label for a platform, used to prefix its tests
function target_label(p)
    str = lowercase(p.name * " " * p.vendor)
    ispocl(p)                ? "pocl"    :
    occursin("nvidia", str)  ? "nvidia"  :
    occursin("intel", str)   ? "intel"   :
    occursin("rusticl", str) ? "rusticl" :
    occursin("amd", str)     ? "amd"     :
    lowercase(first(split(p.vendor)))
end

# per-device label suffix, only when a platform exposes more than one device: "/gpu", "/cpu", plus
# an index when several share a type ("/gpu1", "/gpu2").
function device_suffixes(devices)
    length(devices) <= 1 && return fill("", length(devices))
    types = [string(d.device_type) for d in devices]
    map(enumerate(types)) do (i, t)
        same = findall(==(t), types)
        length(same) > 1 ? "/$t$(findfirst(==(i), same))" : "/$t"
    end
end

# Test targets: each selected device, run through its preferred backend (`:auto`). pocl devices also
# get a `…c` target forcing the OpenCL C source path, where `:auto` would pick SPIR-V IL.
const targets = let ts = []
    for p in OpenCL.cl.platforms()
        platform_selected(p) || continue
        base = target_label(p)
        for (i, suffix) in enumerate(device_suffixes(OpenCL.cl.devices(p)))
            push!(ts, (; label = base * suffix, platform = p.name, index = i, backend = :auto))
            ispocl(p) && push!(ts, (; label = base * "c" * suffix, platform = p.name, index = i, backend = :opencl))
        end
    end
    isempty(ts) && error("No OpenCL platforms found" *
                         (platform_filter === nothing ? "" : " matching '$(platform_filter.value)'"))
    ts
end

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
# wrap a test body on a specific device of the named platform, with a fixed backend
function generate_test(expr, platform_name, device_index, backend)
    return quote
        platform = first(p for p in cl.platforms() if p.name == $platform_name)
        device = cl.devices(platform)[$device_index]
        @testset "$(device.name)" begin
            cl.platform!(platform)
            cl.device!(device)
            OpenCL.program_backend!($(QuoteNode(backend)))
            $(expr)
        end
    end
end

# Duplicate every test per target, prefixing the name so failures and `--` filters name the GPU
# backend (e.g. `pocl/`, `poclc/`, `nvidia/`, or `nvidia/gpu1/` when a platform has several devices).
for name in collect(keys(testsuite))
    body = testsuite[name]
    delete!(testsuite, name)
    for t in targets
        testsuite["$(t.label)/$name"] = generate_test(body, t.platform, t.index, t.backend)
    end
end

## filter
filter_tests!(testsuite, args)

# "indexing scalar" asserts scalar indexing throws when disallowed, which only
# holds for device-only memory; host-accessible memory (SVM, host/shared USM)
# serves it directly. Drop it for targets whose device uses such memory.
if args.list === nothing
    host_accessible = (OpenCL.cl.UnifiedHostMemory, OpenCL.cl.UnifiedSharedMemory,
                       OpenCL.cl.SharedVirtualMemory)
    for t in targets
        plat = first(p for p in OpenCL.cl.platforms() if p.name == t.platform)
        dev = OpenCL.cl.devices(plat)[t.index]
        OpenCL.cl.platform!(plat)
        OpenCL.cl.device!(dev)
        if OpenCL.memory_type() in host_accessible
            delete!(testsuite, "$(t.label)/gpuarrays/indexing scalar")
        end
    end
end

const init_worker_code = quote
    using OpenCL, $pocl_pkg

    OpenCL.allowscalar(false)

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
    import ..@on_device
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
