using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

@clkernel test_ifelse(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if testval % 4 == 0
        v[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfElse") do
    b = cl.Buffer(Bool, ctx, 1)
    test_ifselse_ocl = test_ifselse[queue, (1,)]
    test_ifelse_ocl(b, 10)
    @fact cl.read(queue, b)[1] => false
    test_ifelse_ocl(b, 16)
    @fact cl.read(queue, b)[1] => true
end

