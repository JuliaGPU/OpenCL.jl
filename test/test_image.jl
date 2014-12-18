using FactCheck
using Base.Test

import OpenCL 
const cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

# These tests were adapted from the PyOpenCL library

facts("OpenCL.Image type constructor") do
    @fact @throws_pred(cl.Image{cl.RGBA, Float32}) => (false, "no error")
end

facts("OpenCL.Image supported formats > 0") do
    for device in cl.devices()
	if !(device[:has_image_support])
	    warn("OpenCL.Image not supported on $device")
	    continue
        end
        ctx = cl.Context(device)
	@fact length(cl.supported_image_types(ctx)) > 0 => true
    end
end

const image2d_src = """
 __kernel void copy_image(
	__global float *dest,
	__read_only image2d_t src,
	int stride0)
{
	size_t d0 = get_global_id(0);
	size_t d1 = get_global_id(1);
	
	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE 
		             | CLK_ADDRESS_CLAMP
			     | CLK_FILTER_NEAREST;
        	
	dest[d0 * stride0 + d1] = read_imagef(src, samp, (float2)(d1, d0)).x;
}
"""

facts("OpenCL.Image 2D test") do
    for device in cl.devices()
	if !(device[:has_image_support])
	    warn("OpenCL.Image not supported on $device")
	    continue
        end
        
	ctx   = cl.Context(device)
	queue = cl.CmdQueue(ctx)

	prg  = cl.Program(ctx, source=image2d_src) |> cl.build!
        copy_image = cl.Kernel(prg, "copy_image")

	if !(cl.Image{cl.Red, Float32} in cl.supported_image_types(ctx))
            warn("OpenCL.Image type not supported on $device")
	    continue
        end
	
	a = rand(Float32, (10,10))
	a_img = cl.Image{cl.Red, Float32}(ctx, (:r, :copy), hostbuf=a)
	a_dst = cl.Buffer(Float32, ctx, :rw, length(a))

	x_stride = int32(strides(a)[end])
	evt = copy_image[queue, size(a)](a_dst, a_img, x_stride)

        a_result = reshape(cl.read(queue, a_dst), size(a))
	@fact isapprox(norm(a_result - a), zero(Float32)) => true
    end
end

const image3d_src = """
 __kernel void copy_image(
	__global float2 *dest,
	__read_only image3d_t src,
	int stride0,
	int stride1)
{
	size_t d0 = get_global_id(0);
	size_t d1 = get_global_id(1);
        size_t d2 = get_global_id(2);

	const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE 
		             | CLK_ADDRESS_CLAMP
			     | CLK_FILTER_NEAREST;
	
	dest[d0*stride0 + d1*stride1 + d2] = read_imagef(src, samp, 
							 (float4)(d2, d1, d0,0)).xy;
}
"""

facts("OpenCL.Image 3D test") do
    for device in cl.devices()
	if !(device[:has_image_support])
	    warn("OpenCL.Image not supported on $device")
	    continue
        end
        
	ctx   = cl.Context(device)
	queue = cl.CmdQueue(ctx)

	prg  = cl.Program(ctx, source=image3d_src) |> cl.build!
        copy_image = cl.Kernel(prg, "copy_image")

	if !(cl.Image{cl.Red, Float32} in cl.supported_image_types(ctx))
            warn("OpenCL.Image type not supported on $device")
	    continue
        end
	
	a = rand(Float32, (10,10,10,2))
	a_img = cl.Image{cl.RG, Float32}(ctx, (:r, :copy), hostbuf=a)
	a_dst = cl.Buffer(Float32, ctx, :rw, length(a))

	x_stride = int32(strides(a)[1])
	y_stride = int32(strides(a)[2])
	evt = copy_image[queue, (10,10,10)](a_dst, a_img, x_stride, y_stride)

        a_result = reshape(cl.read(queue, a_dst), size(a))
	@show a
	@show a_result
	@fact isapprox(sum(a_result - a), zero(Float32)) => true
    end
end
