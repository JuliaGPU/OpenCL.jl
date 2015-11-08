import OpenCL.CLArray

facts("OpenCL.CLArray") do

    context("OpenCL.CLArray constructors") do
        for device in cl.devices()

            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            hostarray = zeros(Float32, 32*64)
            A = CLArray(ctx, hostarray)

            @fact CLArray(ctx, queue, (:rw, :copy), hostarray) --> not(nothing) "no error"

            @fact CLArray(ctx, hostarray,
                          queue=queue, flags=(:rw, :copy)) --> not(nothing) "no error"

            @fact CLArray(ctx, hostarray) --> not(nothing) "no error"

            @fact CLArray(cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=hostarray),
                          (32, 64)) --> not(nothing) "no error"

            @fact copy(A) --> A
        end
     end

    context("OpenCL.CLArray fill") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)

            @fact cl.to_host(cl.fill(Float32, queue, @compat(Float32(0.5)),
                                            32, 64)) --> fill(@compat(Float32(0.5)), 32, 64)
            @fact cl.to_host(cl.zeros(Float32, queue, 64)) --> zeros(Float32, 64)
            @fact cl.to_host(cl.ones(Float32, queue, 64)) --> ones(Float32, 64)

        end
     end

    context("OpenCL.CLArray core functions") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            A = CLArray(ctx, rand(Float32, 32, 64))
            @fact size(A) --> (32, 64)
            @fact ndims(A) --> 2
            @fact length(A) --> 32*64
            B = reshape(A, 32*64)
            @fact reshape(B, 32, 64) --> A
        end
     end

    context("OpenCL.CLArray transpose") do
        for device in cl.devices()
            ctx = cl.Context(device)
            queue = cl.CmdQueue(ctx)
            A = CLArray(ctx, rand(Float32, 32, 64))
            B = cl.zeros(Float32, queue, 64, 32)
            Base.transpose!(B, A; block_size=8) 
            @fact cl.to_host(A)' --> cl.to_host(B)            
        end
     end

end
