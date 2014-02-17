using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

@clkernel test_ifelse(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if testval % 4 == 0
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfElse") do
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifelse[queue, (1,)]
    test_ocl(b, 10)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 16)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_ifelseand(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if testval >= 0 && testval < 100
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfElseAnd") do
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifelseand[queue, (1,)]
    test_ocl(b, -1)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 0)
    @fact cl.read(queue, b)[1] => true
    test_ocl(b, 50)
    @fact cl.read(queue, b)[1] => true
    test_ocl(b, 100)
    @fact cl.read(queue, b)[1] => false
end

@clkernel test_ifelseandand(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if (testval % 2 == 0 &&
        testval <= 10 &&
        testval >= 0  &&
        testval % 4 == 0)
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfElseAndAnd") do
    comp = (x) -> (x % 2 == 0 && x <= 10 && x >= 0 && x % 4 == 0) ? true : false
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifelseandand[queue, (1,)]
    for v = -1:101
        test_ocl(b, v)
        @fact cl.read(queue, b)[1] => comp(v)
    end
end


