using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

@clkernel image_convolution(conv_mat::Vector{Float32},
                            img_in::Vector{Uint8}, 
                            img_out::Vector{Uint8},
                            w::Int,
                            h::Int) = begin
    x = get_global_id(0) % (w * 3)
    y = get_global_id(0) / (w * 3)
    if x > 3 && x < (w * 3 - 3) && y > 1 && y < (h - 1)
        accum = 0f0
        count = 0
        for dx = -3:3:3
            for dy = -1:1
                rgb = 0xff * img_in[((y + dy) * w) + (x + dx)]
                accum += rgb * conv_mat[count]
                count += 1
            end
        end
        val = uint8(max(0, min(int(accum), 255)))
        img_out[y * w + x] = val
    end
    return
end

facts("Test example convolution") do
    w = h = 512
    test_arr = ones(Uint8, (w, h))
    conv_ker = [0f0,  -10f0,  0f0,
               -10f0,  40f0, -10f0,
                0f0,  -10f0,  0f0]
    conv_mat = cl.Buffer(Float32, ctx, :copy, hostbuf=conv_ker) 
    img_in   = cl.Buffer(Uint8, ctx, :copy, hostbuf=test_arr)
    img_out  = cl.empty_like(img)

    image_convolution[queue, (w * h,)](conv_mat, img_in, img_out, w, h)
    res = cl.read(queue, img_out)

    @fact true => true
end

