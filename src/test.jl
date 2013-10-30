import OpenCL
cl = OpenCL

println(cl.platforms(), cl.num_platforms())
for p in cl.platforms()
    println(p)
    for k in [:profile, :version, :name, :vendor, :extensions]
        println("\t$k: $(p[k])")
    end
end


platform = cl.platforms()[1]
for p in cl.platforms()
    for k in (:gpu, :cpu, :accelerator, :all)
        println(cl.devices(p, k))
    end
end

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
    println(p)
    for d in cl.devices(p)
        println("\t$d")
        for k in device_info_keys
            println("\t\t$k: $(cl.info(d, k))")
        end
    end
end
