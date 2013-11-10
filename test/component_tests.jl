using FactCheck
using Base.Test

import OpenCL 
cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

# define usable platforms as those that contain
# all working devices
function available_platforms()
    usable_platforms = {}
    for platform in cl.platforms()
        usable = true
        for device in cl.devices(platform)
            try
                cl.Context(device)
            catch err
                usable = false
            end
        end
        if usable
            push!(usable_platforms, platform)
        end
    end
    return usable_platforms
end

immutable TestStruct
    a::cl.CL_int
    b::cl.CL_float
end


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

    context("OpenCL.Context constructor") do
        @fact @throws_pred(cl.Context([])) => (true, "error")
        for platform in cl.platforms()
            for device in cl.devices(platform)
                @fact @throws_pred(cl.Context(device)) => (false, "no error")
            end
        end
    end

    context("OpenCL.Context platform properties") do
        for platform in cl.platforms()
            try
                cl.Context(cl.CL_DEVICE_TYPE_CPU)
            catch err
                @fact typeof(err) => cl.CLError
                @fact err.desc => :CL_INVALID_PLATFORM
            end
            
            if platform[:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Context platform properties for Portable Computing Language Platform")
                continue
            end

            properties = [(cl.CL_CONTEXT_PLATFORM, platform)]
            @fact @throws_pred(cl.Context(cl.CL_DEVICE_TYPE_CPU,
                               properties=properties)) => (false, "no error") 
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
                ctx2 = cl.Context(cl.CL_DEVICE_TYPE_ACCELERATOR,
                                  properties=properties)
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

facts("OpenCL.CmdQueue") do 
    context("OpenCL.CmdQueue device constructor") do
        @fact @throws_pred(cl.CmdQueue(nothing, nothing)) => (true, "error")
        for platform in cl.platforms()
            for device in cl.devices(platform)
                ctx = cl.Context(device)
                @fact @throws_pred(cl.CmdQueue(ctx)) => (false, "no error")
                @fact @throws_pred(cl.CmdQueue(ctx, device)) => (false, "no error")
            end
        end
    end

    context("OpenCL.CmdQueue info") do
        for platform in cl.platforms()
            for device in cl.devices(platform)
                ctx = cl.Context(device)
                for q in (cl.CmdQueue(ctx), cl.CmdQueue(ctx, device))
                    @fact q[:context] => ctx
                    @fact q[:device] => device
                    @fact q[:reference_count] > 0 => true
                    @fact typeof(q[:properties]) => cl.CL_command_queue_properties
                end
            end
        end
    end
end

facts("OpenCL.Event") do
    context("OpenCL.Event status") do
        #TODO: check if this is version 1.2 or greater..
        ctx = cl.create_some_context()
        evt = cl.UserEvent(ctx)
        evt[:status]
        @fact evt[:status] => cl.CL_SUBMITTED
        cl.complete(evt)
        @fact evt[:status] => cl.CL_COMPLETE
    end

    context("OpenCL.Event wait") do
        ctx = cl.create_some_context()
        # create user event
        usr_evt = cl.UserEvent(ctx)
        q = cl.CmdQueue(ctx)
        cl.enqueue_wait_for_events(q, usr_evt)

        # create marker event
        mkr_evt = cl.enqueue_marker(q)
        
        @fact usr_evt[:status] => cl.CL_SUBMITTED
        @fact mkr_evt[:status] => cl.CL_QUEUED

        cl.complete(usr_evt)
        @fact usr_evt[:status] => cl.CL_COMPLETE

        cl.wait(mkr_evt)
        @fact mkr_evt[:status] => cl.CL_COMPLETE
    end

    context("OpenCL.Event callback") do
        for platform in cl.platforms()
            if contains(platform[:name], "AMD")
                msg = "AMD Segfaults on User Event"
                @fact msg => true
                continue
            end
            if contains(platform[:name], "Portable")
                warn("Skipping OpenCL.Event callback for Portable Computing Language Platform")
                continue
            end

            for device in cl.devices(platform)
                callback_called = false

                function test_callback(evt, status) 
                    callback_called = true
                    println("Test Callback") 
                end

                ctx = cl.Context(device)
                usr_evt = cl.UserEvent(ctx)
                queue = cl.CmdQueue(ctx)
                
                cl.enqueue_wait_for_events(queue, usr_evt)
                
                mkr_evt = cl.enqueue_marker(queue)
                cl.add_callback(mkr_evt, test_callback)

                @fact usr_evt[:status] => cl.CL_SUBMITTED
                @fact mkr_evt[:status] => cl.CL_QUEUED
                @fact callback_called => false
                
                cl.complete(usr_evt)
                @fact usr_evt[:status] => cl.CL_COMPLETE
                
                cl.wait(mkr_evt)
                @fact mkr_evt[:status] => cl.CL_COMPLETE
                @fact callback_called => true
            end
        end
    end       
end

facts("OpenCL.Buffer") do

    function create_test_buffer()
        ctx = cl.create_some_context()
        queue = cl.CmdQueue(ctx)
        testarray = zeros(Float32, 1000)
        buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
        return (queue, buf, testarray)
    end

    context("OpenCL.Buffer constructors") do
        ctx = cl.create_some_context()
        testarray = zeros(Float32, 1000)

        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_ONLY,
                                     sizeof(testarray))) => (false, "no error")
        
        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                     sizeof(testarray))) => (false, "no error")
         
        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                     sizeof(testarray))) => (false, "no error")

        buf = cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_WRITE, sizeof(testarray))
        @fact buf.size => sizeof(testarray)

        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_ONLY, 
                                     hostbuf=testarray)) => (false, "no error")

        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                     hostbuf=testarray)) => (false, "no error")

        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                     hostbuf=testarray)) => (false, "no error")
          
        buf = cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_WRITE, hostbuf=testarray)
        @fact buf.size => sizeof(testarray)
        
        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_ONLY,
                                     hostbuf=testarray)) => (false, "no error")

        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                     hostbuf=testarray)) => (false, "no error")

        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                     hostbuf=testarray)) => (false, "no error")

        buf = cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_WRITE, hostbuf=testarray)
        @fact buf.size => sizeof(testarray)
       
        # invalid buffer size should throw error
        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR, +0)) => (true, "error")
        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR, -1)) => (true, "error")

        # invalid flag combinations should throw error
        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_ALLOC_HOST_PTR,
                                     hostbuf=testarray)) => (true, "error")

        # invalid host pointer should throw error
        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR,
                                     hostbuf=C_NULL)) => (true, "error")
        
        @fact @throws_pred(cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR,
                                     hostbuf=C_NULL)) => (true, "error")
     end

     context("OpenCL.Buffer constructors symbols") do
         for device in cl.devices()
             ctx = cl.Context(device)
             
             for mf1 in [:rw, :r, :w]
                 for mf2 in [:copy, :use, :alloc, :null]
                     for mtype in [cl.CL_char,
                                   cl.CL_uchar,
                                   cl.CL_short,
                                   cl.CL_ushort,
                                   cl.CL_int,
                                   cl.CL_uint,
                                   cl.CL_long,
                                   cl.CL_ulong,
                                   cl.CL_half,
                                   cl.CL_float,
                                   cl.CL_double,
                                   #TODO: bool, vector_types, struct_types...
                                   ]
                         testarray = zeros(mtype, 100)
                         if mf2 == :copy || mf2 == :use
                             @fact @throws_pred(cl.Buffer(mtype, ctx, (mf1, mf2), 
                                                          hostbuf=testarray)) => (false, "no error")
                             buf = cl.Buffer(mtype, ctx, (mf1, mf2), hostbuf=testarray)
                             @fact buf.size => sizeof(testarray)
                         elseif mf2 == :alloc
                             @fact @throws_pred(cl.Buffer(mtype, ctx, (mf1, mf2),
                                                          sizeof(testarray))) => (false, "no error")
                             buf = cl.Buffer(mtype, ctx, (mf1, mf2), sizeof(testarray))
                             @fact buf.size => sizeof(testarray)
                         end
                     end
                 end
             end

             #
             test_array = Array(TestStruct, 100)
             @fact @throws_pred(cl.Buffer(TestStruct, ctx, :alloc, sizeof(test_array))) => (false, "no error")
             @fact @throws_pred(cl.Buffer(TestStruct, ctx, :copy, hostbuf=test_array))  => (false, "no error")

             # invalid buffer size should throw error
             @fact @throws_pred(cl.Buffer(Float32, ctx, :alloc, +0)) => (true, "error")
             @fact @throws_pred(cl.Buffer(Float32, ctx, :alloc, -1)) => (true, "error")

             # invalid flag combinations should throw error
             @fact @throws_pred(cl.Buffer(Float32, ctx, (:use, :alloc), 
                                          hostbuf=testarray)) => (true, "error")

             # invalid host pointer should throw error
             @fact @throws_pred(cl.Buffer(Float32, ctx, :copy,
                                          hostbuf=C_NULL)) => (true, "error")
            
             @fact @throws_pred(cl.Buffer(Float32, ctx, :use, 
                                          hostbuf=C_NULL)) => (true, "error")
     
         end
     end

     context("OpenCL.Buffer fill") do
        queue, buf, testarray = create_test_buffer()
        
        @fact buf.size == sizeof(testarray) => true
        cl.fill!(queue, buf, float32(1.0))
        readback = cl.read(queue, buf)
        @fact all(x -> x == 1.0, readback) => true
        @fact all(x -> x == 0.0, testarray) => true
        @fact buf.valid => true
    end

    context("OpenCL.Buffer write!") do
        queue, buf, testarray = create_test_buffer()
        
        @fact buf.size == sizeof(testarray) => true
        cl.write!(queue, buf, ones(Float32, length(testarray)))
        readback = cl.read(queue, buf)
        @fact all(x -> x == 1.0, readback) => true
        @fact buf.valid => true
    end

    context("OpenCL.Buffer empty") do
        ctx = cl.create_some_context()
        testarray = zeros(Float32, 1000)
        buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)

        @fact @throws_pred(cl.empty(Float32, ctx, -1)) => (true, "error") 
        empty_buf = cl.empty(Float32, ctx, 1000)
        @fact empty_buf.size => sizeof(testarray)
        @fact empty_buf.size => buf.size
       
        dims = (100, 100)
        testarray = zeros(Float32, dims)
        empty_buf = cl.empty(Float32, ctx, dims)
        @fact empty_buf.size => sizeof(testarray)
        @fact empty_buf.valid => true
    end
end

facts("OpenCL.Program") do 
    
    test_source = "
    __kernel void sum(__global const float *a,
                      __global const float *b, 
                      __global float *c)
    {
      uint gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    "

    function create_test_program()
        ctx = cl.create_some_context()
        cl.Program(ctx, source=test_source)
    end

    context("OpenCL.Program source constructor") do
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            @fact @throws_pred(cl.Program(ctx, source=test_source)) => (false, "no error")
        end
    end
    
    context("OpenCL.Program info") do
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            
            @fact prg[:context] => ctx
            
            @fact typeof(prg[:devices]) => Vector{cl.Device}
            @fact length(prg[:devices]) > 0 => true 
            @fact device in prg[:devices] => true

            @fact typeof(prg[:source]) => ASCIIString
            @fact prg[:source] => test_source

            #@fact typeof(prg[:binaries]) => Dict{cl.Device, Array{Uint8}}

            @fact prg[:reference_count] > 0 => true
         end
    end

    context("OpenCL.Program build") do 
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            @fact @throws_pred(cl.build!(prg)) => (false, "no error")
            # BUILD_SUCCESS undefined in POCL implementation..
            if device[:platform][:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Program build for Portable Computing Language Platform")
                continue
            end
            @fact prg[:build_status][device] => cl.CL_BUILD_SUCCESS 
            # test build by methods chaining
            @fact prg[:build_status][device] => cl.CL_BUILD_SUCCESS 
        end
    end

    context("OpenCL.Program source code") do
        for device in cl.devices()
           ctx = cl.Context(device)
           prg = cl.Program(ctx, source=test_source)
           @fact prg[:source] => test_source
       end
    end

    context("OpenCL.Program binaries") do
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source) |> cl.build!
            
            @fact device in collect(keys(prg[:binaries])) => true
            binaries = prg[:binaries]
            @fact device in collect(keys(binaries)) => true
            @fact binaries[device] => not(nothing)
            @fact length(binaries[device]) > 0 => true
            prg2 = cl.Program(ctx, binaries=binaries)
            try 
                prg2[:source]
            catch err
                @fact isa(err, cl.CLError) => true
                @fact err.code => -45 
                @fact err.desc => :CL_INVALID_PROGRAM_EXECUTABLE
            end
            @fact prg2[:binaries] == binaries => true
        end
    end
end

facts("OpenCL.Kernel") do 

    test_source = "
    __kernel void sum(__global const float *a,
                      __global const float *b, 
                      __global float *c,
                      const unsigned int count)
    {
      int gid = get_global_id(0);
      if (gid < count) {
          c[gid] = a[gid] + b[gid];
      }
    }
    "

    #TODO: tests for invalid kernel build error && logs...

    context("OpenCL.Kernel constructor") do
        for device in cl.devices()
            if device[:platform][:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Kernel constructor for Portable Computing Language Platform")
                continue
            end
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            @fact @throws_pred(cl.Kernel(prg, "sum")) => (true, "error")
            cl.build!(prg)
            @fact @throws_pred(cl.Kernel(prg, "sum")) => (false, "no error")
        end
    end

    context("OpenCL.Kernel info") do
        for device in cl.devices()
            if device[:platform][:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Kernel info for Portable Computing Language Platform")
                continue
            end
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            cl.build!(prg)
            k = cl.Kernel(prg, "sum")
            @fact k[:name] => "sum"
            @fact k[:num_args] => 4
            @fact k[:reference_count] > 0 => true
            @fact k[:program] => prg
            @fact typeof(k[:attributes]) => ASCIIString
        end
    end 

    context("OpenCL.Kernel mem/workgroup size") do 
        for device in cl.devices()
            if device[:platform][:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Kernel mem/workgroup size for Portable Computing Language Platform")
                continue
            end
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            cl.build!(prg)
            k = cl.Kernel(prg, "sum")
            @fact @throws_pred(cl.private_mem_size(k, device)) => (false, "no error")
            @fact @throws_pred(cl.local_mem_size(k, device)) => (false, "no error")
            @fact @throws_pred(cl.required_work_group_size(k, device)) => (false, "no error")
        end
    end


    context("OpenCL.Kernel set_arg!/set_args!") do
         for device in cl.devices()
            if device[:platform][:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Kernel mem/workgroup size for Portable Computing Language Platform")
                continue
            end

            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            
            prg = cl.Program(ctx, source=test_source) |> cl.build!
            k = cl.Kernel(prg, "sum")

            count  = 1024
            nbytes = count * sizeof(Float32)
            
            A = cl.Buffer(Float32, ctx, :r, nbytes)
            B = cl.Buffer(Float32, ctx, :r, nbytes)
            C = cl.Buffer(Float32, ctx, :w, nbytes)

            # sizeof mem object for buffer in bytes
            @fact sizeof(A.id) => nbytes
            @fact sizeof(B.id) => nbytes
            @fact sizeof(C.id) => nbytes
            
            cl.fill!(queue, A, float32(1.0))
            cl.fill!(queue, B, float32(1.0))
            
            # we use julia's index by one convention
            @fact @throws_pred(cl.set_arg!(k, 1, A))   => (false, "no error")
            @fact @throws_pred(cl.set_arg!(k, 2, B))   => (false, "no error")
            @fact @throws_pred(cl.set_arg!(k, 3, C))   => (false, "no error")
            @fact @throws_pred(cl.set_arg!(k, 4, uint32(count))) => (false, "no error")

            cl.enqueue_kernel(queue, k, count) |> cl.wait
            r = cl.read(queue, C)

            @fact all(x -> x == 2.0, r) => true
            cl.flush(queue)

            # test set_args with new kernel
            k2 = cl.Kernel(prg, "sum")
            cl.set_args!(k2, A, B, C, uint32(count))
            
            cl.enqueue_fill(queue, A, float32(2.0))
            cl.enqueue_fill(queue, B, float32(2.0))
            cl.enqueue_kernel(queue, k, count)
            cl.finish(queue)

            r = cl.read(queue, C)

            @fact all(x -> x == 4.0, r) => true
        end
    end

    context("OpenCL.Kernel enqueue_kernel") do
        for device in cl.devices()
            if device[:platform][:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Kernel mem/workgroup size for Portable Computing Language Platform")
                continue
            end
            
            simple_kernel = "
                __kernel void test(__global float *i) {
                    *i += 1;
                };"
            
            ctx = cl.Context(device)

            h_buff = Float32[1,]
            d_buff = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=h_buff)

            p = cl.Program(ctx, source=simple_kernel) |> cl.build!
            k = cl.Kernel(p, "test")
            q = cl.CmdQueue(ctx)
            
            # blocking call to kernel finishes cmd queue
            cl.call(q, k, 1, 1, d_buff)
            
            r = cl.read(q, d_buff) 
            @fact r[1] => 2
        end
    end
end

#TODO: works when field access is broken out, Array{Float32} does not given consistent alignment
immutable Params
    A::Float32
    B::Float32
    x1::Float32
    x2::Float32
    c::Int32
    Params(a, b, x, c) = begin
        new(float32(a),
            float32(b),
            float32(x[1]),
            float32(x[2]),
            int32(c))
    end
end

const test_struct = "
    typedef struct Params
    {
        float A;
        float B;
        float x[2];  //padding
        int C;
    } Params;


    __kernel void part3(__global const float *a,
                        __global const float *b, 
                        __global float *c,
                        __constant struct Params* test)
    {
        int gid = get_global_id(0);
        c[gid] = test->A * a[gid] + test->B * b[gid] + test->C;
    }
"

facts("OpenCL.Kernel enqueue kernel 2") do
    for device in cl.devices()
        if device[:platform][:name] == "Portable Computing Language"
            warn("Skipping OpenCL.Kernel mem/workgroup size for Portable Computing Language Platform")
            continue
        end

        ctx = cl.Context(device)
        q   = cl.CmdQueue(ctx)
        p   = cl.Program(ctx, source=test_struct) |> cl.build!
        
        part3 = cl.Kernel(p, "part3")
       
        X::Array{Float32} = fill(float32(1.0), 10)
        Y::Array{Float32} = fill(float32(1.0), 10)

        P = [Params(0.5, 10.0, [0.0, 0.0], 3)]
        #TODO: constructor for single immutable types.., chech if passed parameter isbits
        P_buf = cl.Buffer(Params, ctx, :r, sizeof(P))
        cl.write!(q, P_buf, P)
        
        X_buf = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=X)
        Y_buf = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=Y)
        R_buf = cl.Buffer(Float32, ctx, :w, sizeof(X))
        
        global_size = size(X)
        cl.call(q, part3, global_size, nothing, X_buf, Y_buf, R_buf, P_buf)

        r = cl.read(q, R_buf)
        @fact all(x -> x == 13.5, r) => true
    end
end
