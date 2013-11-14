using FactCheck 

import OpenCL 
const cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

immutable TestStruct
    a::cl.CL_int
    b::cl.CL_float
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
        for device in cl.devices()
            ctx = cl.Context(device)
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
            @fact sizeof(buf) => sizeof(testarray)
            
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
        ctx = cl.create_some_context()
        queue = cl.CmdQueue(ctx)
        testarray = zeros(Float32, 1000)
        buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
        return (queue, buf, testarray)
 
     context("OpenCL.Buffer fill") do
        for device in cl.devices()
             ctx = cl.Context(device)
             queue = cl.CmdQueue(ctx)
             testarray = zeros(Float32, 1000)
             buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
             @fact buf.size == sizeof(testarray) => true
             try 
                 cl.fill!(queue, buf, float32(1.0))
                 readback = cl.read(queue, buf)
                 @fact all(x -> x == 1.0, readback) => true
                 @fact all(x -> x == 0.0, testarray) => true
                 @fact buf.valid => true
             catch err
                v = cl.opencl_version(device)
                if v[1] == 1 && v[2] == 2
                    # OpenCL fill defined for all implementations  >= 1.2
                    throw(err)
                end
                info("fill is a OpenCL v1.2 command")
            end
        end
    end

    context("OpenCL.Buffer write!") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            testarray = zeros(Float32, 1000)
            buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
        
            @fact buf.size == sizeof(testarray) => true
            cl.write!(queue, buf, ones(Float32, length(testarray)))
            readback = cl.read(queue, buf)
            @fact all(x -> x == 1.0, readback) => true
            @fact buf.valid => true
        end
    end

    context("OpenCL.Buffer empty") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            testarray = zeros(Float32, 1000)
            buf = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=testarray)
           
            @fact sizeof(cl.empty_like(ctx, buf)) => sizeof(testarray)

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

    context("OpenCL.Buffer copy!") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            test_array = fill(float32(2.0), 1000)
            a_buf = cl.Buffer(Float32, ctx, sizeof(test_array))
            b_buf = cl.Buffer(Float32, ctx, sizeof(test_array))
            c_arr = Array(Float32, size(test_array))
            # host to device buffer
            cl.copy!(queue, a_buf, test_array)
            # device buffer to device buffer
            cl.copy!(queue, b_buf, a_buf)
            # device buffer to host
            cl.copy!(queue, c_arr, b_buf)
            @fact all(x -> isapprox(x, 2.0), c_arr) => true
        end
    end
end
