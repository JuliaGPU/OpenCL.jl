@testset "Kernel" begin
    test_source = "
    __kernel void sum(__global const float *a,
                      __global const float *b,
                      __global float *c,
                      const unsigned int count)
    {
      unsigned int gid = get_global_id(0);
      if (gid < count) {
          c[gid] = a[gid] + b[gid];
      }
    }
    "

    #TODO: tests for invalid kernel build error && logs...

    @testset "constructor" begin
        prg = cl.Program(source=test_source)
        @test_throws ArgumentError cl.Kernel(prg, "sum")
        cl.build!(prg)
        @test cl.Kernel(prg, "sum") != nothing
    end

    @testset "info" begin
        prg = cl.Program(source=test_source)
        cl.build!(prg)
        k = cl.Kernel(prg, "sum")
        @test k.function_name == "sum"
        @test k.num_args == 4
        @test k.reference_count > 0
        @test k.program == prg
        @test typeof(k.attributes) == String
    end

    @testset "mem/workgroup size" begin
        prg = cl.Program(source=test_source)
        cl.build!(prg)
        k = cl.Kernel(prg, "sum")
        wginfo = cl.work_group_info(k, cl.device())
        for sf in [:size, :compile_size, :local_mem_size, :private_mem_size, :prefered_size_multiple]
            @test getproperty(wginfo, sf) != nothing
        end
    end

    @testset "set_arg!/set_args!" begin
        prg = cl.Program(source=test_source) |> cl.build!
        k = cl.Kernel(prg, "sum")

        count  = 1024
        nbytes = count * sizeof(Float32)

        h_ones = ones(Float32, count)

        A = CLArray(h_ones)
        B = CLArray(h_ones)
        C = CLArray{Float32}(undef, count)

        # we use julia's index by one convention
        cl.set_arg!(k, 1, A.data[].mem)
        cl.set_arg!(k, 2, B.data[].mem)
        cl.set_arg!(k, 3, C.data[].mem)
        cl.set_arg!(k, 4, UInt32(count))

        cl.enqueue_kernel(k, count) |> wait
        r = Array(C)

        @test all(x -> x == 2.0, r)
        cl.flush(cl.queue())

        # test set_args with new kernel
        k2 = cl.Kernel(prg, "sum")
        cl.set_args!(k2, A.data[].mem, B.data[].mem, C.data[].mem, UInt32(count))

        h_twos = fill(2f0, count)
        copyto!(A, h_twos)
        copyto!(B, h_twos)

        #TODO: check for ocl version, fill is opencl v1.2
        #cl.enqueue_fill(A, 2f0)
        #cl.enqueue_fill(B, 2f0)

        cl.enqueue_kernel(k, count)

        @test all(x -> x == 4.0, Array(C))
    end

    @testset "clcall" begin
        simple_kernel = "
            __kernel void test(__global float *i) {
                *i += 1;
            };"

        h_buff = Float32[1,]
        d_arr = CLArray(h_buff)

        p = cl.Program(source=simple_kernel) |> cl.build!
        k = cl.Kernel(p, "test")

        # dimensions must be the same size
        @test_throws ArgumentError clcall(k, Tuple{CLPtr{Float32}}, d_arr;
                                          global_size=(1,), local_size=(1,1))
        @test_throws ArgumentError clcall(k, Tuple{CLPtr{Float32}}, d_arr;
                                          global_size=(1,1), local_size=(1,))

        # dimensions are bounded
        max_work_dim = cl.device().max_work_item_dims
        bad = tuple([1 for _ in 1:(max_work_dim + 1)])

        # calls are asynchronous, but cl.read blocks
        clcall(k, Tuple{CLPtr{Float32}}, d_arr)
        @test Array(d_arr) == [2f0]

        # enqueue task is an alias for calling
        # a kernel with a global/local size of 1
        evt = cl.enqueue_task(k)
        @test Array(d_arr) == [3f0]
    end

    @testset "packed structures" begin
        test_source = "
        struct __attribute__((packed)) Test2{
            long f1;
            int __attribute__((aligned (8))) f2;
        };
        __kernel void structest(__global float *out, struct Test2 b){
            out[0] = b.f1;
            out[1] = b.f2;
        }
        "
        prg = cl.Program(source = test_source)
        cl.build!(prg)
        structkernel = cl.Kernel(prg, "structest")
        out = CLArray{Float32}(undef, 2)
        bstruct = (1, Int32(4))
        clcall(structkernel, Tuple{CLPtr{Float32}, Tuple{Int64, Cint}}, out, bstruct)
        @test Array(out) == [1f0, 4f0]
    end

    @testset "vector arguments" begin
        test_source = "
        __kernel void vec3_unpack(__global float *out, float3 a, float3 b) {
            out[0] = a.x;
            out[1] = a.y;
            out[2] = a.z;
            out[3] = b.x;
            out[4] = b.y;
            out[5] = b.z;
        }
        "
        prg = cl.Program(source = test_source)
        cl.build!(prg)
        vec3kernel = cl.Kernel(prg, "vec3_unpack")
        out = CLArray{Float32}(undef, 6)
        # NOTE: the user is responsible for padding the vector to 4 elements
        #       (only on some platforms)
        vec3_a = (1f0, 2f0, 3f0, 0f0)
        vec3_b = (4f0, 5f0, 6f0, 0f0)
        clcall(
            vec3kernel, Tuple{CLPtr{Float32}, NTuple{4, Float32}, NTuple{4, Float32}},
                           out, vec3_a, vec3_b)
        @test Array(out) == [1f0, 2f0, 3f0, 4f0, 5f0, 6f0]
    end
end
