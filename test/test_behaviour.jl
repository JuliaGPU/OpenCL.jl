#=
info(
"======================================================================
                              Running Behavior Tests
      ======================================================================")
=#
@testset "OpenCL Hello World Test" begin

    hello_world_kernel = "
        #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

        __constant char hw[] = \"hello world\";

        __kernel void hello(__global char *out) {
            int tid = get_global_id(0);
            out[tid] = hw[tid];
        }"

    hello_world_str = "hello world"

    for device in cl.devices()
        if device[:platform][:name] == "Portable Computing Language"
            @warn("Skipping OpenCL.Kernel mem/workgroup size for Portable Computing Language Platform")
            continue
        end

        ctx   = cl.Context(device)
        queue = cl.CmdQueue(ctx)

        str_len  = length(hello_world_str) + 1
        out_buf  = cl.Buffer(Cchar, ctx, :w, sizeof(Cchar) * str_len)

        prg   = cl.Program(ctx, source=hello_world_kernel) |> cl.build!
        kern  = cl.Kernel(prg, "hello")

        queue(kern, str_len, nothing, out_buf)
        h = cl.read(queue, out_buf)

        @test cl.CLString(h) == hello_world_str
    end
end


@testset "OpenCL Low Level Api Test" begin

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

    for device in cl.devices()

        len = 1024
        h_a = Vector{cl.CL_float}(undef, len)
        h_b = Vector{cl.CL_float}(undef, len)
        h_c = Vector{cl.CL_float}(undef, len)
        h_d = Vector{cl.CL_float}(undef, len)
        h_e = Vector{cl.CL_float}(undef, len)
        h_f = Vector{cl.CL_float}(undef, len)
        h_g = Vector{cl.CL_float}(undef, len)

        for i in 1:len
            h_a[i] = cl.cl_float(rand())
            h_b[i] = cl.cl_float(rand())
            h_e[i] = cl.cl_float(rand())
            h_g[i] = cl.cl_float(rand())
        end

        err_code = Ref{cl.CL_int}()

        # create compute context (TODO: fails if function ptr's not passed...)
        ctx_id = cl.api.clCreateContext(C_NULL, 1, [device.id],
                                        C_NULL,
                                        C_NULL,
                                        err_code)
        if err_code[] != cl.CL_SUCCESS
            throw(cl.CLError(err_code[]))
        end

        q_id = cl.api.clCreateCommandQueue(ctx_id, device.id, 0, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Failed to create command queue")
        end

        # create program
        bytesource = String(test_source)
        prg_id = cl.api.clCreateProgramWithSource(ctx_id, 1, [bytesource], C_NULL, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Failed to create program")
        end

        # build program
        err = cl.api.clBuildProgram(prg_id, 0, C_NULL, C_NULL, C_NULL, C_NULL)
        if err != cl.CL_SUCCESS
            error("Failed to build program")
        end

        # create compute kernel
        k_id = cl.api.clCreateKernel(prg_id, "sum", err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Failed to create compute kernel")
        end

        # create input array in device memory
        Aid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl.CL_float) * len, h_a, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Error creating buffer A")
        end
        Bid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl.CL_float) * len, h_b, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Error creating buffer B")
        end
        Eid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl.CL_float) * len, h_e, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Error creating buffer E")
        end
        Gid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl.CL_float) * len, h_g, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Error creating buffer G")
        end

        # create output arrays in device memory

        Cid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_READ_WRITE,
                                    sizeof(cl.CL_float) * len, C_NULL, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Error creating buffer C")
        end
        Did = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_READ_WRITE,
                                    sizeof(cl.CL_float) * len, C_NULL, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Error creating buffer D")
        end
        Fid = cl.api.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY,
                                    sizeof(cl.CL_float) * len, C_NULL, err_code)
        if err_code[] != cl.CL_SUCCESS
            error("Error creating buffer F")
        end

        err  = cl.api.clSetKernelArg(k_id, 0, sizeof(cl.CL_mem), [Aid])
        err |= cl.api.clSetKernelArg(k_id, 1, sizeof(cl.CL_mem), [Bid])
        err |= cl.api.clSetKernelArg(k_id, 2, sizeof(cl.CL_mem), [Cid])
        err |= cl.api.clSetKernelArg(k_id, 3, sizeof(cl.CL_uint), cl.CL_uint[len])
        if err != cl.CL_SUCCESS
            error("Error setting kernel 1 args")
        end

        nglobal = Ref{Csize_t}(len)
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
                                         sizeof(cl.CL_float) * len, h_f, 0, C_NULL, C_NULL)
        if err != cl.CL_SUCCESS
            error("Failed to read output array")
        end

        # test results
        for i in 1:len
            tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i]
            @test tmp ≈ h_f[i]
        end
    end
end

struct Params
    A::Float32
    B::Float32
    #TODO: fixed size arrays?
    X1::Float32
    X2::Float32
    C::Int32
    Params(a, b, x, c) = begin
        new(Float32(a),
            Float32(b),
            Float32(x[1]),
            Float32(x[2]),
            Int32(c))
    end
end

let test_struct = "
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

@testset "OpenCL Struct Buffer Test" begin
    for device in cl.devices()

        if device[:platform][:name] == "Portable Computing Language"
            @warn("Skipping OpenCL Struct Buffer Test for Portable Computing Language Platform")
            continue
        end

        ctx = cl.Context(device)
        q   = cl.CmdQueue(ctx)
        p   = cl.Program(ctx, source=test_struct) |> cl.build!

        part3 = cl.Kernel(p, "part3")

        X = fill(1f0, 10)
        Y = fill(1f0, 10)

        P = [Params(0.5, 10.0, [0.0, 0.0], 3)]

        #TODO: constructor for single immutable types.., check if passed parameter isbits
        P_buf = cl.Buffer(Params, ctx, :r, length(P))
        cl.write!(q, P_buf, P)

        X_buf = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=X)
        Y_buf = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=Y)
        R_buf = cl.Buffer(Float32, ctx, :w, length(X))

        global_size = size(X)
        q(part3, global_size, nothing, X_buf, Y_buf, R_buf, P_buf)

        r = cl.read(q, R_buf)
        @test all(x -> x == 13.5, r)
    end
end

end

mutable struct MutableParams
    A::Float32
    B::Float32
end


let test_mutable_pointerfree = "
    typedef struct Params
    {
        float A;
        float B;
    } Params;


    __kernel void part3(
        __global float *a,
        Params test
    ){
        a[0] = test.A;
        a[1] = test.B;
    }
"


@testset "OpenCL Struct Buffer Test" begin
    for device in cl.devices()

        if device[:platform][:name] == "Portable Computing Language"
            @warn("Skipping OpenCL Struct Buffer Test for Portable Computing Language Platform")
            continue
        end

        ctx = cl.Context(device)
        q   = cl.CmdQueue(ctx)
        p   = cl.Program(ctx, source=test_mutable_pointerfree) |> cl.build!

        part3 = cl.Kernel(p, "part3")

        P = MutableParams(0.5, 10.0)
        P_buf = cl.Buffer(Float32, ctx, :w, 2)
        q(part3, 1, nothing, P_buf, P)

        r = cl.read(q, P_buf)

        @test r[1] == 0.5
        @test r[2] == 10.0
    end
end

end
