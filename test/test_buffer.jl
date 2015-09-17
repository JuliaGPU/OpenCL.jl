immutable TestStruct
    a::cl.CL_int
    b::cl.CL_float
end

facts("OpenCL.Buffer") do
    context("OpenCL.Buffer constructors") do
        for device in cl.devices()

            ctx = cl.Context(device)
            testarray = zeros(Float32, 1000)

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_ONLY,
                                         length(testarray)) --> not(nothing) "no error"

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                         length(testarray)) --> not(nothing) "no error"

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                         length(testarray)) --> not(nothing) "no error"

            buf = cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_WRITE, length(testarray))
            @fact buf.len --> length(testarray)

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_ONLY,
                                         hostbuf=testarray) --> not(nothing) "no error"

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                         hostbuf=testarray) --> not(nothing) "no error"

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                         hostbuf=testarray) --> not(nothing) "no error"

            buf = cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_WRITE, hostbuf=testarray)
            @fact buf.len --> length(testarray)

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_ONLY,
                                         hostbuf=testarray) --> not(nothing) "no error"

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                         hostbuf=testarray) --> not(nothing) "no error"

            @fact cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                         hostbuf=testarray) --> not(nothing) "no error"

            buf = cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_WRITE, hostbuf=testarray)
            @fact buf.len --> length(testarray)

            # invalid buffer size should throw error
            @fact_throws cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR, +0) "error"
            @fact_throws cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR, -1) "error"

            # invalid flag combinations should throw error
            @fact_throws cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_ALLOC_HOST_PTR,
                                             hostbuf=testarray) "error"

            # invalid host pointer should throw error
            @fact_throws cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR,
                                         hostbuf=C_NULL) "error"

            @fact_throws cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR,
                                         hostbuf=C_NULL) "error"
        end
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
                             @fact cl.Buffer(mtype, ctx, (mf1, mf2), hostbuf=testarray) --> not(nothing) "no error"
                             buf = cl.Buffer(mtype, ctx, (mf1, mf2), hostbuf=testarray)
                             @fact buf.len --> length(testarray)
                         elseif mf2 == :alloc
                             @fact cl.Buffer(mtype, ctx, (mf1, mf2),
                                                          length(testarray)) --> not(nothing) "no error"
                             buf = cl.Buffer(mtype, ctx, (mf1, mf2), length(testarray))
                             @fact buf.len --> length(testarray)
                         end
                     end
                 end
             end

             test_array = Array(TestStruct, 100)
             @fact cl.Buffer(TestStruct, ctx, :alloc, length(test_array)) --> not(nothing) "no error"
             @fact cl.Buffer(TestStruct, ctx, :copy, hostbuf=test_array) --> not(nothing) "no error"

             # invalid buffer size should throw error
             @fact_throws cl.Buffer(Float32, ctx, :alloc, +0) "error"
             @fact_throws cl.Buffer(Float32, ctx, :alloc, -1) "error"

             # invalid flag combinations should throw error
             @fact_throws cl.Buffer(Float32, ctx, (:use, :alloc), hostbuf=test_array) "error"

             # invalid host pointer should throw error
             @fact_throws cl.Buffer(Float32, ctx, :copy, hostbuf=C_NULL) "error"

             @fact_throws cl.Buffer(Float32, ctx, :use, hostbuf=C_NULL) "error"

         end
     end

     context("OpenCL.Buffer fill") do
        for device in cl.devices()
             if contains(device[:platform][:name], "Portable")
                 # the pocl platform claims to implement v1.2 of the spec, but does not
                 warn("Skipping test OpenCL.Buffer fill for POCL Platform")
                 continue
             end
             ctx = cl.Context(device)
             queue = cl.CmdQueue(ctx)
             testarray = zeros(Float32, 1000)
             buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
             @fact buf.len == length(testarray) --> true

             v = cl.opencl_version(device)
             if v.major == 1 && v.minor < 2
                 platform_name = device[:platform][:name]
                 info("Skipping OpenCL.Buffer fill for $platform_name: fill is a v1.2 command")
                 continue
             end
             cl.fill!(queue, buf, 1f0)
             readback = cl.read(queue, buf)
             @fact all(x -> x == 1.0, readback) --> true
             @fact all(x -> x == 0.0, testarray) --> true
             @fact buf.valid --> true
        end
    end

    context("OpenCL.Buffer write!") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            testarray = zeros(Float32, 1000)
            buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
            @fact buf.len == length(testarray) --> true
            cl.write!(queue, buf, ones(Float32, length(testarray)))
            readback = cl.read(queue, buf)
            @fact all(x -> x == 1.0, readback) --> true
            @fact buf.valid --> true
        end
    end

    context("OpenCL.Buffer empty_like") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            testarray = zeros(Float32, 1000)
            buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)

            @fact sizeof(cl.empty_like(ctx, buf)) --> sizeof(testarray)
        end
    end

    context("OpenCL.Buffer copy!") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            test_array = fill(2f0, 1000)
            a_buf = cl.Buffer(Float32, ctx, length(test_array))
            b_buf = cl.Buffer(Float32, ctx, length(test_array))
            c_arr = Array(Float32, length(test_array))
            # host to device buffer
            cl.copy!(queue, a_buf, test_array)
            # device buffer to device buffer
            cl.copy!(queue, b_buf, a_buf)
            # device buffer to host
            cl.copy!(queue, c_arr, b_buf)
            @fact all(x -> isapprox(x, 2.0), c_arr) --> true
        end
    end

    context("OpenCL.Buffer map/unmap") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            b = cl.Buffer(Float32, ctx, :rw, 100)
            for f in (:r, :w, :rw)
                a, evt = cl.enqueue_map_mem(queue, b, f, 0, (10,10))
                cl.wait(evt)
                @fact size(a) --> (10,10)
                @fact typeof(a) --> Array{Float32,2}

                # cannot unmap a buffer without same host array
                bad = similar(a)
                @fact_throws cl.unmap!(queue, b, bad) "error"

                @fact cl.ismapped(b) --> true
                cl.unmap!(queue, b, a)
                @fact cl.ismapped(b) --> false

                # cannot unmap an unmapped buffer
                @fact_throws cl.unmap!(queue, b, a) "error"

                # gc here quickly force any memory errors
                Base.gc()
            end
            @fact cl.ismapped(b) --> false
            a, evt = cl.enqueue_map_mem(queue, b, :rw, 0, (10,10))
            @fact cl.ismapped(b) --> true
            evt = cl.enqueue_unmap_mem(queue, b, a, wait_for=evt)
            cl.wait(evt)
            @fact cl.ismapped(b) --> false
        end
    end
end
