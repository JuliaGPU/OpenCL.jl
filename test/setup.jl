#=
using IOCapture

using Random


## entry point

const targets = []

function runtests(f, name, platform_filter)


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
                    startswith(name, "gpuarrays/") || startswith(name, "device/")


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

 =#