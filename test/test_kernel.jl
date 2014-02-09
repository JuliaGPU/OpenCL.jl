using FactCheck
using Base.Test

import OpenCL 
const cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

facts("OpenCL.Kernel") do 

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

    context("OpenCL.Kernel constructor") do
        for device in cl.devices()
            if device[:platform][:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Kernel constructor for " * 
                     "Portable Computing Language Platform")
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
            for (sf, clf) in [(:size, cl.CL_KERNEL_WORK_GROUP_SIZE),
                              (:compile_size, cl.CL_KERNEL_COMPILE_WORK_GROUP_SIZE),
                              (:local_mem_size, cl.CL_KERNEL_LOCAL_MEM_SIZE),
                              (:private_mem_size, cl.CL_KERNEL_PRIVATE_MEM_SIZE),
                              (:prefered_size_multiple, cl.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)]
                @fact @throws_pred(cl.work_group_info(k, sf, device)) => (false, "no error")
                @fact @throws_pred(cl.work_group_info(k, clf, device)) => (false, "no error")
                if sf != :compile_size
                    @fact cl.work_group_info(k, sf, device) => cl.work_group_info(k, clf, device)
                end
            end
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
           
            h_ones = ones(Float32, count)

            A = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=h_ones)
            B = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=h_ones)
            C = cl.Buffer(Float32, ctx, :w, count)

            # sizeof mem object for buffer in bytes
            @fact sizeof(A) => nbytes
            @fact sizeof(B) => nbytes
            @fact sizeof(C) => nbytes
            
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
            
            h_twos = fill(float32(2.0), count)
            cl.copy!(queue, A, h_twos)
            cl.copy!(queue, B, h_twos)
            
            #TODO: check for ocl version, fill is opencl v1.2
            #cl.enqueue_fill(queue, A, float32(2.0))
            #cl.enqueue_fill(queue, B, float32(2.0))
            
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
           
            # dimensions must be the same size
            @fact @throws_pred(cl.call(q, k, (1,), (1,1), d_buff)) => (true, "error")
            @fact @throws_pred(cl.call(q, k, (1,1), (1,), d_buff)) => (true, "error")

            # dimensions are bounded
            max_work_dim = device[:max_work_item_dims]
            bad = tuple([1 for _ in 1:(max_work_dim + 1)])
            @fact @throws_pred(cl.call(q, k, bad, d_buff)) => (true, "error")

            # devices have finite work sizes
            @fact @throws_pred(cl.call(q, k, (typemax(Int),) d_buff)) => (true, "error")

            # blocking call to kernel finishes cmd queue
            cl.call(q, k, 1, 1, d_buff)
            
            r = cl.read(q, d_buff) 
            @fact r[1] => 2

            # alternative kernel call syntax
            k[q, (1,), (1,)](d_buff)
            r = cl.read(q, d_buff)
            @fact r[1] => 3 

            # enqueue task is an alias for calling 
            # a kernel with a global/local size of 1
            evt = cl.enqueue_task(q, k)
            r = cl.read(q, d_buff)
            @fact r[1] => 4
        end
    end
end
