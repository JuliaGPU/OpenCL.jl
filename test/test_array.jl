import OpenCL.CLArray

facts("OpenCL.CLArray") do

    context("OpenCL.CLArray constructors") do
        for device in cl.devices()

            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            hostarray = zeros(Float32, 128*64)
            A = CLArray(ctx, hostarray)

            @fact CLArray(ctx, queue, (:rw, :copy), hostarray) --> not(nothing) "no error"

            @fact CLArray(ctx, hostarray,
                          queue=queue, flags=(:rw, :copy)) --> not(nothing) "no error"

            @fact CLArray(ctx, hostarray) --> not(nothing) "no error"

            @fact CLArray(cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=hostarray),
                          (128, 64)) --> not(nothing) "no error"

            @fact copy(A) --> A
        end
     end

    context("OpenCL.CLArray fill") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)

            @fact cl.to_host(cl.fill(Float32, queue, Float32(0.5),
                                            32, 64)) --> fill(Float32(0.5), 32, 64)
            @fact cl.to_host(cl.zeros(Float32, queue, 64)) --> zeros(Float32, 64)
            @fact cl.to_host(cl.ones(Float32, queue, 64)) --> ones(Float32, 64)

        end
     end

    context("OpenCL.CLArray core functions") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            A = CLArray(ctx, rand(Float32, 128, 64))
            @fact size(A) --> (128, 64)
            @fact ndims(A) --> 2
            @fact length(A) --> 128*64
            B = reshape(A, 128*64)
            @fact reshape(B, 128, 64) --> A
        end
     end

    context("OpenCL.CLArray transpose") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            A = CLArray(ctx, rand(Float32, 32, 64))
            B = cl.zeros(Float32, queue, 64, 32)
            transpose!(B, A) 
            @fact cl.to_host(A') --> cl.to_host(B)            
        end
     end

end
