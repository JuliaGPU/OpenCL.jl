@testset "Device" begin
    @testset "Type" begin
        for (t, k) in zip((cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_CPU,
                           cl.CL_DEVICE_TYPE_ACCELERATOR, cl.CL_DEVICE_TYPE_ALL),
                          (:gpu, :cpu, :accelerator, :all))

            #for (dk, dt) in zip(cl.devices(cl.platform(), k), cl.devices(cl.platform(), t))
            #    @fact dk == dt --> true
            #end
            #devices = cl.devices(cl.platform(), k)
            #for device in devices
            #    @fact device.device_type == t --> true
            #end
        end
    end

    @testset "Equality" begin
        devices = cl.devices(cl.platform())

        if length(devices) > 1
            d1 = devices[1]
            for d2 in devices[2:end]
               @test pointer(d2) != pointer(d1)
               @test hash(d2) != hash(d1)
               @test isequal(d2, d1) == false
           end
       end
    end

    @testset "Info" begin
        device_info_keys = Symbol[
                :driver_version,
                :version,
                :extensions,
                :platform,
                :name,
                :device_type,
                :has_image_support,
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
        @test isa(cl.platform(), cl.Platform)
        @test_throws ErrorException cl.platform().zjdlkf

        device = cl.device()
        @test isa(device, cl.Device)
        @test_throws ErrorException device.zjdlkf
        for k in device_info_keys
            v = getproperty(device, k)
            if k == :extensions
                @test isa(v, Array)
                if length(v) > 0
                    @test isa(v, Array{String, 1})
                end
            elseif k == :platform
                @test v == cl.platform()
            elseif k == :max_work_item_sizes
                @test length(v) == 3
            elseif k == :max_image2d_shape
                @test length(v) == 2
            elseif k == :max_image3d_shape
                @test length(v) == 3
            end
        end

        @test cl.queue_properties(cl.device()).profiling isa Bool
        @test cl.queue_properties(cl.device()).out_of_order_exec isa Bool

        @test cl.exec_capabilities(cl.device()).native_kernel isa Bool

        @test cl.svm_capabilities(cl.device()).fine_grain_buffer isa Bool
    end
end
