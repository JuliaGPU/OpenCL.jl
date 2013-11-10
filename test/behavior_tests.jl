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



facts("OpenCL.Kernel low level api") do
    for device in cl.devices()
        
        length = 1024
        h_a = Array(cl.CL_float, length)
        h_b = Array(cl.CL_float, length)
        h_c = Array(cl.CL_float, length)
        h_d = Array(cl.CL_float, length)
        h_e = Array(cl.CL_float, length)
        h_f = Array(cl.CL_float, length)
        h_g = Array(cl.CL_float, length)

        for i in 1:length
            h_a[i] = cl.cl_float(rand())
            h_b[i] = cl.cl_float(rand())
            h_e[i] = cl.cl_float(rand())
            h_g[i] = cl.cl_float(rand())
        end 
        
        err_code = Array(cl.CL_int, 1)

        # create compute context (TODO: fails if function ptr's not passed...)
        ctx_id = cl.api.clCreateContext(C_NULL, 1, [device.id], 
                                        cl.ctx_callback_ptr, 
                                        cl.raise_context_error, 
                                        err_code)
        if err_code[1] != cl.CL_SUCCESS
            #error("Failed to create compute context")
            throw(cl.CLError(err_code[1]))
        end

        q_id = cl.api.clCreateCommandQueue(ctx_id, device.id, 0, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Failed to create command queue")
        end

        # create program
        bytesource = bytestring(test_source)
        prg_id = cl.api.clCreateProgramWithSource(ctx_id, 1, [bytesource], C_NULL, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Failed to create program")
        end

        # build program
        err = cl.api.clBuildProgram(prg_id, 0, C_NULL, C_NULL, C_NULL, C_NULL)
        if err != cl.CL_SUCCESS
            error("Failed to build program")
        end
        
        # create compute kernel
        k_id = cl.api.clCreateKernel(prg_id, "sum", err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Failed to create compute kernel")
        end

        # create input array in device memory
        Aid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl.CL_float) * length, h_a, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Error creating buffer A")
        end
        Bid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl.CL_float) * length, h_b, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Error creating buffer B")
        end
        Eid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl.CL_float) * length, h_e, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Error creating buffer E")
        end
        Gid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl.CL_float) * length, h_g, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Error creating buffer G")
        end

        # create output arrays in device memory

        Cid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_READ_WRITE,
                                    sizeof(cl.CL_float) * length, C_NULL, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Error creating buffer C")
        end
        Did = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_READ_WRITE, 
                                    sizeof(cl.CL_float) * length, C_NULL, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Error creating buffer D")
        end
        Fid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY, 
                                    sizeof(cl.CL_float) * length, C_NULL, err_code)
        if err_code[1] != cl.CL_SUCCESS
            error("Error creating buffer F")
        end

        err  = cl.api.clSetKernelArg(k_id, 0, sizeof(cl.CL_mem), [Aid])
        err |= cl.api.clSetKernelArg(k_id, 1, sizeof(cl.CL_mem), [Bid])
        err |= cl.api.clSetKernelArg(k_id, 2, sizeof(cl.CL_mem), [Cid])
        err |= cl.api.clSetKernelArg(k_id, 3, sizeof(cl.CL_uint), cl.CL_uint[length])
        if err != cl.CL_SUCCESS
            error("Error setting kernel 1 args")
        end
        
        nglobal = Csize_t[length,]
        err = cl.api.clEnqueueNDRangeKernel(q_id, k_id,  1, C_NULL,
                                            nglobal, C_NULL, 0, C_NULL, C_NULL)
        if err != cl.CL_SUCCESS
            error("Failed to execute kernel 1")
        end

        err  = cl.api.clSetKernelArg(k_id, 0, sizeof(cl.CL_mem), [Eid])
        err |= cl.api.clSetKernelArg(k_id, 1, sizeof(cl.CL_mem), [Cid])
        err |= cl.api.clSetKernelArg(k_id, 2, sizeof(cl.CL_mem), [Did])
        if err != cl.CL_SUCCESS
            error("Error setting kernel 2 args")
        end
        err = cl.api.clEnqueueNDRangeKernel(q_id, k_id,  1, C_NULL,
                                            nglobal, C_NULL, 0, C_NULL, C_NULL)
        if err != cl.CL_SUCCESS
            error("Failed to execute kernel 2")
        end

        err  = cl.api.clSetKernelArg(k_id, 0, sizeof(cl.CL_mem), [Gid])
        err |= cl.api.clSetKernelArg(k_id, 1, sizeof(cl.CL_mem), [Did])
        err |= cl.api.clSetKernelArg(k_id, 2, sizeof(cl.CL_mem), [Fid])
        if err != cl.CL_SUCCESS
            error("Error setting kernel 3 args")
        end
        err = cl.api.clEnqueueNDRangeKernel(q_id, k_id,  1, C_NULL,
                                            nglobal, C_NULL, 0, C_NULL, C_NULL)
        if err != cl.CL_SUCCESS
            error("Failed to execute kernel 3")
        end

        # read back the result from compute device...
        err = cl.api.clEnqueueReadBuffer(q_id, Fid, cl.CL_TRUE, 0,
                                         sizeof(cl.CL_float) * length, h_f, 0, C_NULL, C_NULL)
        if err != cl.CL_SUCCESS
            error("Failed to read output array")
        end

        # test results
        ncorrect = 0
        for i in 1:length
            tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i]
            if isapprox(tmp, h_f[i])
                ncorrect += 1
            end
        end
        @fact ncorrect => length
    end
end

#TODO: works when field access is broken out, Array{Float32} does not given consistent alignment
immutable Params
    A::Float32
    B::Float32
    X1::Float32
    X2::Float32
    C::Int32
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
       
        X = fill(float32(1.0), 10)
        Y = fill(float32(1.0), 10)

        P = [Params(0.5, 10.0, [0.0, 0.0], 3)]
        
        #TODO: constructor for single immutable types.., check if passed parameter isbits
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
