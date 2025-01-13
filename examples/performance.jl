using OpenCL, Printf, LinearAlgebra

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
        c_temp = c_temp * (a_temp/2.0f); // times 1/2 my a

        c[gid] = c_temp; // store result in global memory
}"

function cl_performance(ndatapts::Integer, nworkers::Integer)

    @assert ndatapts > 0
    @assert nworkers > 0

    a = rand(Float32,  ndatapts)
    b = rand(Float32,  ndatapts)
    c = Vector{Float32}(undef, ndatapts)

    @printf("Size of test data: %i MB\n", sizeof(a) / 1024 / 1024)

    t1 = time()
    for i in 1:ndatapts
        c_temp = a[i] + b[i]
        c_temp = c_temp * c_temp
        c[i]   = c_temp * (a[i] / 2f0)
    end
    t2 = time()

    @printf("Julia Execution time: %.4f seconds\n", t2 - t1)

    for platform in cl.platforms()
        cl.platform!(platform)

        for device in cl.available_devices(platform)
            cl.device!(device)

            @printf("====================================================\n")
            @printf("Platform name:    %s\n",  platform.name)
            @printf("Platform profile: %s\n",  platform.profile)
            @printf("Platform vendor:  %s\n",  platform.vendor)
            @printf("Platform version: %s\n",  platform.version)
            @printf("----------------------------------------------------\n")
            @printf("Device name: %s\n", device.name)
            @printf("Device type: %s\n", device.device_type)
            @printf("Device mem: %i MB\n",           device.global_mem_size / 1024^2)
            @printf("Device max mem alloc: %i MB\n", device.max_mem_alloc_size / 1024^2)
            @printf("Device max clock freq: %i MHZ\n",  device.max_clock_frequency)
            @printf("Device max compute units: %i\n",   device.max_compute_units)
            @printf("Device max work group size: %i\n", device.max_work_group_size)
            @printf("Device max work item size: %s\n",  device.max_work_item_size)

            if device.max_mem_alloc_size < sizeof(Float32) * ndatapts
                @warn("Requested buffer size exceeds device max alloc size!")
                @warn("Skipping device $(device.name)...")
                continue
            end

            if device.max_work_group_size < nworkers
                @warn("Number of workers exceeds the device's max work group size!")
                @warn("Skipping device $(device.name)...")
                continue
            end

            da = CLArray(a; access=:r)
            db = CLArray(b; access=:r)
            dc = CLArray{Float32}(undef, length(a); access=:w)

            prg  = cl.Program(source=bench_kernel) |> cl.build!
            kern = cl.Kernel(prg, "sum")

            # work_group_multiple = kern.prefered_work_group_size_multiple
            global_size = (ndatapts,)
            local_size  = (nworkers,)

            cl.queue!(:profile) do
                # call the kernel
                evt = clcall(kern, Tuple{Ptr{Float32}, Ptr{Float32}, Ptr{Float32}},
                             da, db, dc; global_size, local_size)
                wait(evt)

                # duration in ns
                t = evt.profile_duration * 1e-9
                @printf("Execution time of test: %.4f seconds\n", t)

                @info("Result norm: $(norm(c - Array(dc)))")
            end
        end
    end
end

# Play with these numbers to see performance differences
# N_DATAPTS has to be a multiple of the number of workers
# N_WORKERS has to be less than or equal to the device's max work group size
# ex. N_WORKERS = 1 is non parallel execution on the gpu

const N_DATA_PTS = Int(2^23) # ~8 million
const N_WORKERS  = Int(2^7)
cl_performance(N_DATA_PTS, N_WORKERS)
