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

            @fact CLArray(ctx, hostarray; queue=queue) --> not(nothing) "no error"

            @fact CLArray(cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=hostarray),
                          (128, 64); queue=queue) --> not(nothing) "no error"

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
            A = CLArray(ctx, rand(Float32, 128, 64); queue=queue)
            @fact size(A) --> (128, 64)
            @fact ndims(A) --> 2
            @fact length(A) --> 128*64
            # reshape
            B = reshape(A, 128*64)
            @fact reshape(B, 128, 64) --> A
            # transpose
            X = CLArray(ctx, rand(Float32, 32, 32); queue=queue)
            B = cl.zeros(Float32, queue, 64, 128)
            # on Travis in a build for Mac, MAX_WORK_ITEM_SIZE is equal (1024, 1, 1)
            # while transpose's default block_size is 32, so skipping this test for Mac
            if get(ENV, "TRAVIS", "false") != "true" || (@linux ? true : false)
                ev = transpose!(B, A)
                cl.wait(ev)
                @fact cl.to_host(A') --> cl.to_host(B)
            end
        end
     end

end
