using FactCheck
using Base.Test

import OpenCL
cl = OpenCL

facts("OpenCL.Kernel hello world") do
    
    hello_world_kernel = "
        #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
        
        __constant char hw[] = \"hello world\";

        __kernel void hello(__global char *out) {
            size_t tid = get_global_id(0);
            out[tid] = hw[tid];
        }"

    hello_world_str = "hello world"
    
    for device in [cl.devices()[1]]
        if device[:platform][:name] == "Portable Computing Language"
            warn("Skipping OpenCL.Kernel mem/workgroup size for Portable Computing Language Platform")
            continue
        end
        ctx   = cl.Context(device)
        queue = cl.CmdQueue(ctx)
        
        h_len  = length(hello_world_str) + 1 
        out_h  = Array(cl.CL_char, h_len) 
        #out_cl = cl.Buffer(ctx,
        #                   cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_USE_HOST_PTR,
        #                   hostbuf=out_h)

        nbytes    = sizeof(cl.CL_char) * h_len
        out_cl_id = cl._create_cl_buffer(ctx.id, cl.CL_MEM_WRITE_ONLY,  
                                         cl.cl_uint(nbytes), C_NULL)
        out_cl = cl.Buffer{cl.CL_char}(out_cl_id, false, cl.cl_uint(nbytes))

        prg   = cl.Program(ctx, source=hello_world_kernel) |> cl.build!
        kern  = cl.Kernel(prg, "hello")
        cl.set_arg!(kern, 1, out_cl)
        evt_id = cl.api.clEnqueueNDRangeKernel(queue.id, kern.id,
                                               cl.uint(1),
                                               C_NULL,
                                               Csize_t[h_len,],
                                               Csize_t[1,1,],
                                               cl.cl_uint(0),
                                               C_NULL, C_NULL)
        cl.wait(Event(evt_id))
        h = cl.read(queue, out_cl)
        @fact bytestring(convert(Ptr{Char}, h)) => hello_world_str
    end
end

facts("OpenCL.Kernel simple kernel") do
    for device in [cl.devices()[1]]
        if device[:platform][:name] == "Portable Computing Language"
            warn("Skipping OpenCL.Kernel mem/workgroup size for Portable Computing Language Platform")
            continue
        end
        ctx = cl.Context(device)
        queue = cl.CmdQueue(ctx)
        
        simple_kernel = "
            __kernel void test(__global float *i) {
                *i += 1;
            }"

        prg = cl.Program(ctx, source=simple_kernel) |> cl.build!
        k   = cl.Kernel(prg, "test")
        
        nbytes = sizeof(Float32) * 1
        A = cl.Buffer(ctx, cl.cl_mem_flags(0), nbytes)
        cl.write!(queue, A, Float32[1,])
        @fact cl.read(queue, A) => Float32[1,]
        @fact sizeof(A.id) => sizeof(Float32[1,])
        @fact k[:reference_count] > 0 => true
        @fact cl.reference_count(A) => 1
        println(cl.reference_count(A))
        cl.set_arg!(k, 1, A)
        try
            evt = cl.enqueue_kernel(queue, k, 1)
            #cl.wait(evt)
        catch err
            println("error: $device")
            throw(err)
        end
        #@fact cl.read(queue, A) => Float32[2,] 
    end
end

