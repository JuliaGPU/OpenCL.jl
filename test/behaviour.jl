#=
info(
"======================================================================
                              Running Behavior Tests
      ======================================================================")
=#
@testset "Hello World Test" begin
    hello_world_kernel = "
        #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

        __constant char hw[] = \"hello world\";

        __kernel void hello(__global char *out) {
            int tid = get_global_id(0);
            out[tid] = hw[tid];
        }"

    hello_world_str = "hello world"


    ctx   = cl.Context(device)
    queue = cl.CmdQueue(ctx)

    str_len  = length(hello_world_str) + 1
    out_buf  = cl.Buffer(Cchar, ctx, sizeof(Cchar) * str_len, :w)

    prg   = cl.Program(ctx, source=hello_world_kernel) |> cl.build!
    kern  = cl.Kernel(prg, "hello")

    queue(kern, str_len, nothing, out_buf)
    h = cl.read(queue, out_buf)

    @test cl.CLString(h) == hello_world_str
end

@testset "Low Level API Test" begin

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

    len = 1024
    h_a = Vector{Cfloat}(undef, len)
    h_b = Vector{Cfloat}(undef, len)
    h_c = Vector{Cfloat}(undef, len)
    h_d = Vector{Cfloat}(undef, len)
    h_e = Vector{Cfloat}(undef, len)
    h_f = Vector{Cfloat}(undef, len)
    h_g = Vector{Cfloat}(undef, len)

    for i in 1:len
        h_a[i] = Cfloat(rand())
        h_b[i] = Cfloat(rand())
        h_e[i] = Cfloat(rand())
        h_g[i] = Cfloat(rand())
    end

    err_code = Ref{cl.Cint}()

    # create compute context (TODO: fails if function ptr's not passed...)
    ctx_id = cl.clCreateContext(C_NULL, 1, [device.id],
                                    C_NULL,
                                    C_NULL,
                                    err_code)
    if err_code[] != cl.CL_SUCCESS
        throw(cl.CLError(err_code[]))
    end

    q_id = cl.clCreateCommandQueue(ctx_id, device.id, 0, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Failed to create command queue")
    end

    # create program
    bytesource = String(test_source)
    prg_id = cl.clCreateProgramWithSource(ctx_id, 1, [bytesource], C_NULL, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Failed to create program")
    end

    # build program
    cl.clBuildProgram(prg_id, 0, C_NULL, C_NULL, C_NULL, C_NULL)

    # create compute kernel
    k_id = cl.clCreateKernel(prg_id, "sum", err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Failed to create compute kernel")
    end

    # create input array in device memory
    Aid = cl.clCreateBuffer(ctx_id, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                sizeof(Cfloat) * len, h_a, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Error creating buffer A")
    end
    Bid = cl.clCreateBuffer(ctx_id, cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                sizeof(Cfloat) * len, h_b, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Error creating buffer B")
    end
    Eid = cl.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                sizeof(Cfloat) * len, h_e, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Error creating buffer E")
    end
    Gid = cl.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                                sizeof(Cfloat) * len, h_g, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Error creating buffer G")
    end

    # create output arrays in device memory

    Cid = cl.clCreateBuffer(ctx_id, cl.CL_MEM_READ_WRITE,
                                sizeof(Cfloat) * len, C_NULL, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Error creating buffer C")
    end
    Did = cl.clCreateBuffer(ctx_id, cl.CL_MEM_READ_WRITE,
                                sizeof(Cfloat) * len, C_NULL, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Error creating buffer D")
    end
    Fid = cl.clCreateBuffer(ctx_id, cl.CL_MEM_WRITE_ONLY,
                                sizeof(Cfloat) * len, C_NULL, err_code)
    if err_code[] != cl.CL_SUCCESS
        error("Error creating buffer F")
    end

    cl.clSetKernelArg(k_id, 0, sizeof(cl.cl_mem), [Aid])
    cl.clSetKernelArg(k_id, 1, sizeof(cl.cl_mem), [Bid])
    cl.clSetKernelArg(k_id, 2, sizeof(cl.cl_mem), [Cid])
    cl.clSetKernelArg(k_id, 3, sizeof(cl.Cuint), cl.Cuint[len])

    nglobal = Ref{Csize_t}(len)
    cl.clEnqueueNDRangeKernel(q_id, k_id,  1, C_NULL,
                                  nglobal, C_NULL, 0, C_NULL, C_NULL)

    cl.clSetKernelArg(k_id, 0, sizeof(cl.cl_mem), [Eid])
    cl.clSetKernelArg(k_id, 1, sizeof(cl.cl_mem), [Cid])
    cl.clSetKernelArg(k_id, 2, sizeof(cl.cl_mem), [Did])
    cl.clEnqueueNDRangeKernel(q_id, k_id,  1, C_NULL,
                                  nglobal, C_NULL, 0, C_NULL, C_NULL)

    cl.clSetKernelArg(k_id, 0, sizeof(cl.cl_mem), [Gid])
    cl.clSetKernelArg(k_id, 1, sizeof(cl.cl_mem), [Did])
    cl.clSetKernelArg(k_id, 2, sizeof(cl.cl_mem), [Fid])
    cl.clEnqueueNDRangeKernel(q_id, k_id,  1, C_NULL,
                                  nglobal, C_NULL, 0, C_NULL, C_NULL)

    # read back the result from compute device...
    cl.clEnqueueReadBuffer(q_id, Fid, cl.CL_TRUE, 0,
                               sizeof(Cfloat) * len, h_f, 0, C_NULL, C_NULL)

    # test results
    for i in 1:len
        tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i]
        @test tmp ≈ h_f[i]
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

@testset "Struct Buffer Test" begin
    ctx = cl.Context(device)
    q   = cl.CmdQueue(ctx)
    p   = cl.Program(ctx, source=test_struct) |> cl.build!

    part3 = cl.Kernel(p, "part3")

    X = fill(1f0, 10)
    Y = fill(1f0, 10)

    P = [Params(0.5, 10.0, [0.0, 0.0], 3)]

    #TODO: constructor for single immutable types.., check if passed parameter isbits
    P_buf = cl.Buffer(Params, ctx, length(P), :r)
    cl.write!(q, P_buf, P)

    X_buf = cl.Buffer(Float32, ctx, length(X), (:r, :copy), hostbuf=X)
    Y_buf = cl.Buffer(Float32, ctx, length(Y), (:r, :copy), hostbuf=Y)
    R_buf = cl.Buffer(Float32, ctx, length(X), :w)

    global_size = size(X)
    q(part3, global_size, nothing, X_buf, Y_buf, R_buf, P_buf)

    r = cl.read(q, R_buf)
    @test all(x -> x == 13.5, r)
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


@testset "Struct Buffer Test" begin
    ctx = cl.Context(device)
    q   = cl.CmdQueue(ctx)
    p   = cl.Program(ctx, source=test_mutable_pointerfree) |> cl.build!

    part3 = cl.Kernel(p, "part3")

    P = MutableParams(0.5, 10.0)
    P_buf = cl.Buffer(Float32, ctx, 2, :w)
    q(part3, 1, nothing, P_buf, P)

    r = cl.read(q, P_buf)

    @test r[1] == 0.5
    @test r[2] == 10.0
end

end
