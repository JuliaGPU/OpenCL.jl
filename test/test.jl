using FactCheck
using Base.Test

import OpenCL 
cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

facts("OpenCL.Platform") do 
    
    context("Platform Info") do
        @fact length(cl.platforms()) => cl.num_platforms()
        for p in cl.platforms()
            @fact p != nothing => true
            @fact pointer(p) != C_NULL => true
            for k in [:profile, :version, :name, :vendor, :extensions]
                @fact p[k] == cl.info(p, k) => true
            end
         end
     end
     
     context("Platform Equality") do 
        platform       = cl.platforms()[1]
        platform_copy  = cl.platforms()[1]
        
        @fact pointer(platform) => pointer(platform_copy) 
        @fact hash(platform) => hash(platform_copy)
        @fact isequal(platform, platform) => true
        
        if length(cl.platforms()) > 1
            for p in cl.platforms()[2:end]
                @fact pointer(platform) == pointer(p) => false
                @fact hash(platform) == hash(p) => false
                @fact isequal(platform, p) => false
            end
        end
    end
end

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
                :max_work_item_sizes,
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
                :max_workgroup_size,
                :max_parameter_size,
                :profiling_timer_resolution,
                :max_image2d_shape,
                :max_image3d_shape,
            ]
        for p in cl.platforms()
            @fact isa(p, cl.Platform) => true
            @fact @throws_pred(p[:zjdlkf]) => (true, "error")
            for d in cl.devices(p)
                @fact isa(d, cl.Device) => true
                @fact @throws_pred(d[:zjdlkf]) => (true, "error")
                for k in device_info_keys
                    @fact @throws_pred(d[k]) => (false, "no error")
                    @fact d[k] => cl.info(d, k)
                    if k == :extensions
                        @fact isa(d[k], Vector{String}) => true 
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

facts("OpenCL.Context") do

    context("OpenCL.Context device constructor") do
        platform = cl.platforms()[1]
        device = cl.devices(platform)[1]
        @fact @throws_pred(cl.Context(device)) => (false, "no error")
        @fact @throws_pred(cl.Context([])) => (true, "error")
    end

    context("OpenCL.Context device_type constructor") do
        for platform in cl.platforms()
            try
                cl.Context(cl.CL_DEVICE_TYPE_CPU)
            catch err
                @fact typeof(err) => cl.CLError
                @fact err.desc => :CL_INVALID_PLATFORM
            end
            properties = [(cl.CL_CONTEXT_PLATFORM, platform)]
            @fact @throws_pred(cl.Context(cl.CL_DEVICE_TYPE_CPU, properties=properties)) => (false, "no error") 
            ctx = cl.Context(cl.CL_DEVICE_TYPE_CPU, properties=properties)
            @fact isempty(cl.properties(ctx)) => false
            test_properties = cl.properties(ctx)
            platform_in_properties = false 
            for (t, v) in test_properties
                if t == cl.CL_CONTEXT_PLATFORM
                    @fact v[:name] => platform[:name]
                    @fact v == platform => true
                    platform_in_properties = true
                    break
                end
            end
            @fact platform_in_properties => true 
            @fact @throws_pred(cl.Context(:cpu, properties=properties)) => (false, "no error")
            try
                ctx2 = cl.Context(cl.CL_DEVICE_TYPE_ACCELERATOR, properties=properties)
            catch err
                @fact typeof(err) => cl.CLError
                @fact err.desc => :CL_DEVICE_NOT_FOUND
            end
        end
    end

    context("OpenCL.Context create_some_context") do
        @fact @throws_pred(cl.create_some_context()) => (false, "no error")
        @fact typeof(cl.create_some_context()) => cl.Context
    end
end


