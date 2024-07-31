@testset "Device" begin
    @testset "Type" begin
        for (t, k) in zip((cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_CPU,
                           cl.CL_DEVICE_TYPE_ACCELERATOR, cl.CL_DEVICE_TYPE_ALL),
                          (:gpu, :cpu, :accelerator, :all))

            #for (dk, dt) in zip(cl.devices(platform, k), cl.devices(platform, t))
            #    @fact dk == dt --> true
            #end
            #devices = cl.devices(platform, k)
            #for device in devices
            #    @fact device[:device_type] == t --> true
            #end
        end
    end

    @testset "Equality" begin
        devices = cl.devices(platform)

        if length(devices) > 1
            d1 = devices[1]
            for d2 in devices[2:end]
               @test pointer(d2) != pointer(d1)
               @test hash(d2) != hash(d1)
               @test isequal(d2, d1) == false
           end
       end
    end

    if occursin("Portable", platform[:name])
        @warn("Skipping Device Info tests for Portable Computing Language Platform")
    else
        @testset "Info" begin
            device_info_keys = Symbol[
                    :driver_version,
                    :version,
                    :extensions,
                    :platform,
                    :name,
                    :device_type,
                    :has_image_support,
                    :queue_properties,
                    :has_queue_out_of_order_exec,
                    :has_queue_profiling,
                    :has_native_kernel,
                    :vendor_id,
                    :max_compute_units,
                    :max_work_item_size,
                    :max_clock_frequency,
                    :address_bits,
                    :max_read_image_args,
                    :max_write_image_args,
                    :global_mem_size,
                    :max_mem_alloc_size,
                    :max_const_buffer_size,
                    :local_mem_size,
                    :has_local_mem,
                    :host_unified_memory,
                    :available,
                    :compiler_available,
                    :max_work_group_size,
                    :max_parameter_size,
                    :profiling_timer_resolution,
                    :max_image2d_shape,
                    :max_image3d_shape,
                ]
            @test isa(platform, cl.Platform)
            @test_throws ArgumentError platform[:zjdlkf]

            @test isa(device, cl.Device)
            @test_throws ArgumentError device[:zjdlkf]
            for k in device_info_keys
                @test device[k] == cl.info(device, k)
                if k == :extensions
                    @test isa(device[k], Array)
                    if length(device[k]) > 0
                        @test isa(device[k], Array{String, 1})
                    end
                elseif k == :platform
                    @test device[k] == platform
                elseif k == :max_work_item_sizes
                    @test length(device[k]) == 3
                elseif k == :max_image2d_shape
                    @test length(device[k]) == 2
                elseif k == :max_image3d_shape
                    @test length(device[k]) == 3
                end
            end
        end
    end
end
