struct CLTestStruct
    f1::NTuple{3, Float32}
    f2::Nothing
    f3::Float32
end

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
        prg = cl.Program(cl.context(), source=test_source)
        @test_throws ArgumentError cl.Kernel(prg, "sum")
        cl.build!(prg)
        @test cl.Kernel(prg, "sum") != nothing
    end

    @testset "info" begin
        prg = cl.Program(cl.context(), source=test_source)
        cl.build!(prg)
        k = cl.Kernel(prg, "sum")
        @test k[:name] == "sum"
        @test k[:num_args] == 4
        @test k[:reference_count] > 0
        @test k[:program] == prg
        @test typeof(k[:attributes]) == String
    end

    @testset "mem/workgroup size" begin
        prg = cl.Program(cl.context(), source=test_source)
        cl.build!(prg)
        k = cl.Kernel(prg, "sum")
        for (sf, clf) in [(:size, cl.CL_KERNEL_WORK_GROUP_SIZE),
                          (:compile_size, cl.CL_KERNEL_COMPILE_WORK_GROUP_SIZE),
                          (:local_mem_size, cl.CL_KERNEL_LOCAL_MEM_SIZE),
                          (:private_mem_size, cl.CL_KERNEL_PRIVATE_MEM_SIZE),
                          (:prefered_size_multiple, cl.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)]
            @test cl.work_group_info(k, sf, cl.device()) != nothing
            @test cl.work_group_info(k, clf, cl.device()) != nothing
            if sf != :compile_size
                @test cl.work_group_info(k, sf, cl.device()) == cl.work_group_info(k, clf, cl.device())
            end
        end
    end

    @testset "set_arg!/set_args!" begin
        prg = cl.Program(cl.context(), source=test_source) |> cl.build!
        k = cl.Kernel(prg, "sum")

        count  = 1024
        nbytes = count * sizeof(Float32)

        h_ones = ones(Float32, count)

        A = cl.Buffer(Float32, length(h_ones), (:r, :copy), hostbuf=h_ones)
        B = cl.Buffer(Float32, length(h_ones), (:r, :copy), hostbuf=h_ones)
        C = cl.Buffer(Float32, count, :w)

        # sizeof mem object for buffer in bytes
        @test sizeof(A) == nbytes
        @test sizeof(B) == nbytes
        @test sizeof(C) == nbytes

        # we use julia's index by one convention
        @test cl.set_arg!(k, 1, A) != nothing
        @test cl.set_arg!(k, 2, B) != nothing
        @test cl.set_arg!(k, 3, C) != nothing
        @test cl.set_arg!(k, 4, UInt32(count)) != nothing

        cl.enqueue_kernel(cl.queue(), k, count) |> cl.wait
        r = cl.read(C)

        @test all(x -> x == 2.0, r)
        cl.flush(cl.queue())

        # test set_args with new kernel
        k2 = cl.Kernel(prg, "sum")
        cl.set_args!(k2, A, B, C, UInt32(count))

        h_twos = fill(2f0, count)
        cl.copy!(A, h_twos)
        cl.copy!(B, h_twos)

        #TODO: check for ocl version, fill is opencl v1.2
        #cl.enqueue_fill(cl.queue(), A, 2f0)
        #cl.enqueue_fill(cl.queue(), B, 2f0)

        cl.enqueue_kernel(cl.queue(), k, count)
        cl.finish(cl.queue())

        r = cl.read(C)

        @test all(x -> x == 4.0, r)
    end

    @testset "enqueue_kernel" begin
        simple_kernel = "
            __kernel void test(__global float *i) {
                *i += 1;
            };"

        h_buff = Float32[1,]
        d_buff = cl.Buffer(Float32, length(h_buff), (:rw, :copy), hostbuf=h_buff)

        p = cl.Program(cl.context(), source=simple_kernel) |> cl.build!
        k = cl.Kernel(p, "test")

        # dimensions must be the same size
        @test_throws ArgumentError cl.queue()(k, (1,), (1,1), d_buff)
        @test_throws ArgumentError cl.queue()(k, (1,1), (1,), d_buff)

        # dimensions are bounded
        max_work_dim = cl.device()[:max_work_item_dims]
        bad = tuple([1 for _ in 1:(max_work_dim + 1)])
        @test_throws MethodError cl.queue()(k, bad, d_buff)

        # devices have finite work sizes
        @test_throws MethodError cl.queue()(k, (typemax(Int),), d_buff)

        # blocking call to kernel finishes cmd queue
        cl.queue()(k, 1, 1, d_buff)

        r = cl.read(d_buff)
        @test r[1] == 2

        # alternative kernel call syntax
        k[cl.queue(), (1,), (1,)](d_buff)
        r = cl.read(d_buff)
        @test r[1] == 3

        # enqueue task is an alias for calling
        # a kernel with a global/local size of 1
        evt = cl.enqueue_task(cl.queue(), k)
        r = cl.read(d_buff)
        @test r[1] == 4
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
        prg = cl.Program(cl.context(), source = test_source)
        cl.build!(prg)
        structkernel = cl.Kernel(prg, "structest")
        out = cl.Buffer(Float32, 2, :w)
        bstruct = (1, Int32(4))
        structkernel[cl.queue(), (1,)](out, bstruct)
        r = cl.read(out)
        @test r  == [1f0, 4f0]
    end

    @testset "empty types" begin
        test_source = "
        //packed
        struct __attribute__((packed)) Test{
            float3 f1;
            int f2; // empty type gets replaced with Int32 (no empty types allowed in OpenCL)
            // you might need to define the alignement of fields to match julia's layout
            float f3; // for the types used here the alignement matches though!
        };
        __kernel void structest(__global float *out, struct Test a){
            out[0] = a.f1.x;
            out[1] = a.f1.y;
            out[2] = a.f1.z;
            out[3] = a.f3;
        }
        "

        prg = cl.Program(cl.context(), source = test_source)
        cl.build!(prg)
        structkernel = cl.Kernel(prg, "structest")
        out = cl.Buffer(Float32, 4, :w)
        astruct = CLTestStruct((1f0, 2f0, 3f0), nothing, 22f0)
        structkernel[cl.queue(), (1,)](out, astruct)
        r = cl.read(out)
        @test r == [1f0, 2f0, 3f0, 22f0]
    end
end
