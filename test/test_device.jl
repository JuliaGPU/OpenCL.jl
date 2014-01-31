using FactCheck

import OpenCL 
const cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

facts("OpenCL.Device") do 
    
    context("Device Type") do
        for p in cl.platforms()
            for (t, k) in zip((cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_CPU, 
                               cl.CL_DEVICE_TYPE_ACCELERATOR, cl.CL_DEVICE_TYPE_ALL), 
                              (:gpu, :cpu, :accelerator, :all))
                
                #for (dk, dt) in zip(cl.devices(p, k), cl.devices(p, t))
                #    @fact dk == dt => true
                #end
                #devices = cl.devices(p, k)
                #for d in devices
                #    @fact d[:device_type] == t => true
                #end
            end
        end
    end

    context("Device Equality") do
        for platform in cl.platforms()
            devices = cl.devices(platform)
            if length(devices) > 1
                test_dev = devices[1]
                for dev in devices[2:end]
                   @fact pointer(dev) != pointer(test_dev) => true
                   @fact hash(dev) != hash(test_dev) => true
                   @fact isequal(dev, test_dev) => false
               end
           end
       end

    end

    context("Device Info") do 
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
        for p in cl.platforms()
            if contains(p[:name], "Portable")
                msg = "Skipping Device Info tests for Portable Computing Language Platform "
                warn(msg)
                continue
            end
            @fact isa(p, cl.Platform) => true
            @fact @throws_pred(p[:zjdlkf]) => (true, "error")
            for d in cl.devices(p)
                @fact isa(d, cl.Device) => true
                @fact @throws_pred(d[:zjdlkf]) => (true, "error")
                for k in device_info_keys
                    @fact @throws_pred(d[k]) => (false, "no error")
                    @fact d[k] => cl.info(d, k)
                    if k == :extensions
                        @fact isa(d[k], Array) => true 
                        if length(d[k]) > 0
                            @fact isa(d[k], Array{String, 1}) => true
                        end
                    elseif k == :platform
                        @fact d[k] => p 
                    elseif k == :max_work_item_sizes
                        @fact length(d[k]) => 3
                    elseif k == :max_image2d_shape
                        @fact length(d[k]) => 2
                    elseif k == :max_image3d_shape
                        @fact length(d[k]) => 3
                    end
                end
            end
        end
    end
end


