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

end
