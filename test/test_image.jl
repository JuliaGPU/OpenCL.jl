using FactCheck
using Base.Test

import OpenCL 
const cl = OpenCL

facts("Nonempty supported formats") do
    for device in cl.devices()
	if !(device[:has_image_support])
	    warn("OpenCL.Image not supported on $device")
	    continue
        end
        ctx = cl.Context(device)
	@test length(cl.supported_image_types(ctx)) > 0
    end
end
