import OpenCL
const cl = OpenCL

const bench_kernel = "
__kernel void sum(__global const float *a, 
                  __global const float *b,
                  __global float *c)
{
        int gid = get_global_id(0);
        float a_temp;
        float b_temp;
        float c_temp;

        a_temp = a[gid]; // my a element (by global ref)
        b_temp = b[gid]; // my b element (by global ref)
        
        c_temp = a_temp+b_temp; // sum of my elements
        c_temp = c_temp * c_temp; // product of sums
        c_temp = c_temp * (a_temp/2.0); // times 1/2 my a

        c[gid] = c_temp; // store result in global memory
}"

function cl_performance(ndatapts::Integer, nworkers::Integer)
    
    @assert ndatapts > 0
    @assert nworkers > 0

    a = rand(Float32,  ndatapts)
    b = rand(Float32,  ndatapts)
    c = Array(Float32, ndatapts)
    
    @printf("Size of test data: %i MB\n", sizeof(a) / 1024 / 1024)

    t1 = time()
    for i in 1:ndatapts
        c_temp = a[i] + b[i]
        c_temp = c_temp * c_temp
        c[i]   = c_temp * (a[i] / float32(2.0))
    end
    t2 = time()

    @printf("Julia Execution time: %.4f seconds\n", t2 - t1)

    for platform in cl.platforms()

        if platform[:name] == "Portable Computing Language"
            warn("Portable Computing Language platform not yet supported")
            continue
        end

        for device in cl.devices(platform)
            @printf("====================================================\n")
            @printf("Platform name:    %s\n",  platform[:name])
            @printf("Platform profile: %s\n",  platform[:profile])
            @printf("Platform vendor:  %s\n",  platform[:vendor])
            @printf("Platform version: %s\n",  platform[:version])
            @printf("----------------------------------------------------\n")
            @printf("Device name: %s\n", device[:name])
            @printf("Device type: %s\n", device[:device_type])
            @printf("Device mem: %i MB\n",           device[:global_mem_size] / 1024^2)
            @printf("Device max mem alloc: %i MB\n", device[:max_mem_alloc_size] / 1024^2)
            @printf("Device max clock freq: %i MHZ\n",  device[:max_clock_frequency])
            @printf("Device max compute units: %i\n",   device[:max_compute_units])
            @printf("Device max work group size: %i\n", device[:max_work_group_size])
            @printf("Device max work item size: %s\n",  device[:max_work_item_size])

            if device[:max_mem_alloc_size] <= sizeof(Float32) * ndatapts
                warn("Requested buffer size exceeds device max alloc size!")
                warn("Skipping device $(device[:name])...")
                continue
            end

            ctx   = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            
            a_buf = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=a)
            b_buf = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=b)
            c_buf = cl.Buffer(Float32, ctx, :w, sizeof(a))

            prg  = cl.Program(ctx, source=bench_kernel) |> cl.build!
            kern = cl.Kernel(prg, "sum")

            # work_group_multiple = kern[:prefered_work_group_size_multiple]
            
            t1 = time()
            #TODO: this does not work in local size is scalar..
            cl.call(queue, kern, (ndatapts,), (nworkers,), a_buf, b_buf, c_buf)
            t2 = time()

            @printf("Execution time of test: %.4f seconds\n", t2 - t1)

            c_device = cl.read(queue, c_buf)
            info("Result norm: $(norm(c - c_device))")
        end
    end
end

cl_performance(int(2^25), 256)

