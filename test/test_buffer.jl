immutable TestStruct
    a::cl.CL_int
    b::cl.CL_float
end

@testset "OpenCL.Buffer" begin
    @testset "OpenCL.Buffer constructors" begin
        for device in cl.devices()

            ctx = cl.Context(device)
            testarray = zeros(Float32, 1000)

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_ONLY,
                                         length(testarray)) != nothing

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                         length(testarray)) != nothing

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                         length(testarray)) != nothing

            buf = cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR | cl.CL_MEM_READ_WRITE, length(testarray))
            @test length(buf) == length(testarray)

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_ONLY,
                                         hostbuf=testarray) != nothing

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                         hostbuf=testarray) != nothing

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                         hostbuf=testarray) != nothing

            buf = cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR | cl.CL_MEM_READ_WRITE, hostbuf=testarray)
            @test length(buf) == length(testarray)

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_ONLY,
                                         hostbuf=testarray) != nothing

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_WRITE_ONLY,
                                         hostbuf=testarray) != nothing

            @test cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_WRITE,
                                         hostbuf=testarray) != nothing

            buf = cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_READ_WRITE, hostbuf=testarray)
            @test length(buf) == length(testarray)

            # invalid buffer size should throw error
            @test_throws cl.CLError cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR, +0)
            @test_throws InexactError cl.Buffer(Float32, ctx, cl.CL_MEM_ALLOC_HOST_PTR, -1)

            # invalid flag combinations should throw error
            @test_throws cl.CLError cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR | cl.CL_MEM_ALLOC_HOST_PTR,
                                             hostbuf=testarray)

            # invalid host pointer should throw error
            @test_throws TypeError cl.Buffer(Float32, ctx, cl.CL_MEM_COPY_HOST_PTR,
                                         hostbuf=C_NULL)

            @test_throws TypeError cl.Buffer(Float32, ctx, cl.CL_MEM_USE_HOST_PTR,
                                         hostbuf=C_NULL)
        end
     end

     @testset "OpenCL.Buffer constructors symbols" begin
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
                             @test cl.Buffer(mtype, ctx, (mf1, mf2), hostbuf=testarray) != nothing
                             buf = cl.Buffer(mtype, ctx, (mf1, mf2), hostbuf=testarray)
                             @test length(buf) == length(testarray)
                         elseif mf2 == :alloc
                             @test cl.Buffer(mtype, ctx, (mf1, mf2),
                                                          length(testarray)) != nothing
                             buf = cl.Buffer(mtype, ctx, (mf1, mf2), length(testarray))
                             @test length(buf) == length(testarray)
                         end
                     end
                 end
             end

             test_array = Vector{TestStruct}(100)
             @test cl.Buffer(TestStruct, ctx, :alloc, length(test_array)) != nothing
             @test cl.Buffer(TestStruct, ctx, :copy, hostbuf=test_array) != nothing

             # invalid buffer size should throw error
             @test_throws cl.CLError cl.Buffer(Float32, ctx, :alloc, +0)
             @test_throws InexactError cl.Buffer(Float32, ctx, :alloc, -1)

             # invalid flag combinations should throw error
             @test_throws ArgumentError cl.Buffer(Float32, ctx, (:use, :alloc), hostbuf=test_array)

             # invalid host pointer should throw error
             @test_throws TypeError cl.Buffer(Float32, ctx, :copy, hostbuf=C_NULL)

             @test_throws TypeError cl.Buffer(Float32, ctx, :use, hostbuf=C_NULL)

         end
     end

     @testset "OpenCL.Buffer fill" begin
        for device in cl.devices()
             if occursin("Portable", device[:platform][:name])
                 # the pocl platform claims to implement v1.2 of the spec, but does not
                 warn("Skipping test OpenCL.Buffer fill for POCL Platform")
                 continue
             end
             ctx = cl.Context(device)
             queue = cl.CmdQueue(ctx)
             testarray = zeros(Float32, 1000)
             buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
             @test length(buf) == length(testarray)

             v = cl.opencl_version(device)
             if v.major == 1 && v.minor < 2
                 platform_name = device[:platform][:name]
                 info("Skipping OpenCL.Buffer fill for $platform_name: fill is a v1.2 command")
                 continue
             end
             cl.fill!(queue, buf, 1f0)
             readback = cl.read(queue, buf)
             @test all(x -> x == 1.0, readback)
             @test all(x -> x == 0.0, testarray)
             @test buf.valid == true
        end
    end

    @testset "OpenCL.Buffer write!" begin
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            testarray = zeros(Float32, 1000)
            buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
            @test length(buf) == length(testarray)
            cl.write!(queue, buf, ones(Float32, length(testarray)))
            readback = cl.read(queue, buf)
            @test all(x -> x == 1.0, readback) == true
            @test buf.valid == true
        end
    end

    @testset "OpenCL.Buffer empty_like" begin
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            testarray = zeros(Float32, 1000)
            buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)

            @test sizeof(cl.empty_like(ctx, buf)) == sizeof(testarray)
        end
    end

    @testset "OpenCL.Buffer copy!" begin
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            test_array = fill(2f0, 1000)
            a_buf = cl.Buffer(Float32, ctx, length(test_array))
            b_buf = cl.Buffer(Float32, ctx, length(test_array))
            c_arr = Vector{Float32}(length(test_array))
            # host to device buffer
            cl.copy!(queue, a_buf, test_array)
            # device buffer to device buffer
            cl.copy!(queue, b_buf, a_buf)
            # device buffer to host
            cl.copy!(queue, c_arr, b_buf)
            @test all(x -> isapprox(x, 2.0), c_arr) == true
        end
    end

    @testset "OpenCL.Buffer map/unmap" begin
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            b = cl.Buffer(Float32, ctx, :rw, 100)
            for f in (:r, :w, :rw)
                a, evt = cl.enqueue_map_mem(queue, b, f, 0, (10,10))
                cl.wait(evt)
                @test size(a) == (10,10)
                @test typeof(a) == Array{Float32,2}

                # cannot unmap a buffer without same host array
                bad = similar(a)
                @test_throws ArgumentError cl.unmap!(queue, b, bad)

                @test cl.ismapped(b) == true
                cl.unmap!(queue, b, a)
                @test cl.ismapped(b) == false

                # cannot unmap an unmapped buffer
                @test_throws ArgumentError cl.unmap!(queue, b, a)

                # gc here quickly force any memory errors
                Base.gc()
            end
            @test cl.ismapped(b) == false
            a, evt = cl.enqueue_map_mem(queue, b, :rw, 0, (10,10))
            @test cl.ismapped(b) == true
            evt = cl.enqueue_unmap_mem(queue, b, a, wait_for=evt)
            cl.wait(evt)
            @test cl.ismapped(b) == false
        end
    end
end
