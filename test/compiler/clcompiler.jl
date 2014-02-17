using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

@clkernel test_for(b::Vector{Bool}) = begin
    gid = get_global_id(0)
    for i=0:(10-1)
        b[i] = true
    end
    return
end

facts("Test For") do
    testbuf = zeros(Bool, 20)
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=testbuf)
    test_ocl = test_for[queue, (1,)]
    test_ocl(b)
    res = cl.read(queue, b)
    @fact all(res[1:10]) => true
    @fact any(res[11:end]) => false
end

@clkernel test_forbreak(b::Vector{Bool}) = begin
    gid = get_global_id(0)
    for i=0:(20-1)
        if i == 10
            break
        end
        b[i] = true
    end
    return
end

facts("Test ForBreak") do
    testbuf = zeros(Bool, 20)
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=testbuf)
    test_ocl = test_forbreak[queue, (1,)]
    test_ocl(b)
    res = cl.read(queue, b)
    @fact all(res[1:10]) => true
    @fact any(res[11:end]) => false
end

@clkernel test_forif(b::Vector{Bool}) = begin
    gid = get_global_id(0)
    for i=0:(20-1)
        if i == 10
            b[i] = true
        end
    end
    return
end

facts("Test ForIf") do
    testbuf = zeros(Bool, 20)
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=testbuf)
    test_ocl = test_forif[queue, (1,)]
    test_ocl(b)
    res = cl.read(queue, b)
    @fact res[11] => true
end

@clkernel test_if(b::Vector{Bool}, testval::Int) = begin 
    gid = get_global_id(0)
    if testval % 4 == 0
        b[gid] = true
    end
    return
end

facts("Test If") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_if[queue, (1,)]
    test_ocl(b, 10)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 12)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_ifand(b::Vector{Bool}, testval::Int) = begin 
    gid = get_global_id(0)
    if testval >= 0 && testval < 100
        b[gid] = true
    end
    return 
end

facts("Test IfAnd") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_ifand[queue, (1,)]
    test_ocl(b, -1)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 100)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 1)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_ifandand(b::Vector{Bool}, testval::Int) = begin 
    gid = get_global_id(0)
    if testval >= 0 && testval < 100 && testval == 20
        b[gid] = true
    end
    return 
end

facts("Test IfAndAnd") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_ifandand[queue, (1,)]
    test_ocl(b, 1) 
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 20) 
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_ifandandand(b::Vector{Bool}, testval::Int) = begin 
    gid = get_global_id(0)
    if ((testval % 2)==0 && testval<=10 && testval>=0 && (testval % 4)==0)
         b[gid] = true
    end
    return
end

facts("Test IfAndAndAnd") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_ifandandand[queue, (1,)]
    test_ocl(b, 13)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 4)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_ifandorand(b::Vector{Bool}, testval::Int) = begin 
    gid = get_global_id(0)
    if ((testval % 2)==0 && testval <=10 || testval>=0 && (testval % 4)==0)
         b[gid] = true
    end
    return
end

facts("Test IfAndOrAnd") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_ifandorand[queue, (1,)]
    test_ocl(b, 5)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 4)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_boolandandand(res::Vector{Bool}) = begin 
    gid = get_global_id(0)
    a = b = c = d = true
    if a && b && c && d
        res[gid] = true
    end
    return
end 

facts("Test BoolAndAndAnd") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_boolandandand[queue, (1,)]
    test_ocl(b)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_boolandandor(res::Vector{Bool}) = begin 
    gid = get_global_id(0)
    a = b = c = d = true
    if a && b && c || d
        res[gid] = true
    end
    return
end 

facts("Test BoolAndAndOr") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_boolandandor[queue, (1,)]
    test_ocl(b)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_boolandorand(res::Vector{Bool}) = begin 
    gid = get_global_id(0)
    a = b = c = d = true
    if (a && b) || (c && d)
        res[gid] = true
    end
    return
end 

facts("Test BoolAndOrAnd") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_boolandandor[queue, (1,)]
    test_ocl(b)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_boolororor(res::Vector{Bool}) = begin 
    gid = get_global_id(0)
    a = b = c = d = true
    if a || b || c || d
        res[gid] = true
    end 
    return
end

facts("Test BoolOrOrOr") do
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=[false])
    test_ocl = test_boolandandor[queue, (1,)]
    test_ocl(b)
    @fact cl.read(queue, b)[1] => true
end

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

#TODO: bool is not allowed as function arg, convert to char when compiling
@clkernel test_ifelseifelseifelse(b::Vector{Bool}, x::Int, y::Int, z::Int) = begin
    gid = get_global_id(0)
    if x
        b[gid] = true
    elseif y
        b[gid+1] = true
    elseif z
        b[gid+2] = true
    else
        b[gid+3] = true
    end
    return
end

facts("Test IfElseAndAnd") do
    #TODO: make it easy to create an array of zeros
    b = cl.Buffer(Bool, ctx, :copy, hostbuf=zeros(Bool, 4))
    test_ocl = test_ifelseifelseifelse[queue, (1,)]
    test_ocl(b, 0, 0, 0)
    @fact cl.read(queue, b) => [false, false, false, true]
    test_ocl(b, 0, 0, 1)
    @fact cl.read(queue, b) => [false, false, true, true]
    test_ocl(b, 0, 1, 1)
    @fact cl.read(queue, b) => [false, true, true, true]
    test_ocl(b, 1, 1, 1)
    @fact cl.read(queue, b) => [true, true, true, true]
end

@clkernel test_ifelsenot_oror_and(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if (testval % 2) != 0 && 
        testval > 0 && 
        testval < 100 || 
       (testval % 4) !=0 
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfElseNot_OrOr_And") do
    comp = (testval) -> begin
        if (testval % 2) != 0 && 
            testval > 0 && 
            testval < 100 || 
           (testval % 4) !=0 
           true
       else
           false
       end
    end
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifelsenot_oror_and[queue, (1,)]
    for v = -1:101
        test_ocl(b, v)
        @fact cl.read(queue, b)[1] => comp(v)
    end
end

@clkernel test_ifelse_oror_and(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if (testval % 2) == 0 || 
        testval <= 0 || 
        testval >= 100 && 
       (testval % 4) == 0
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfElse_OrOr_And") do
    comp = (testval) -> begin
       if (testval % 2) == 0 || 
           testval <= 0 || 
           testval >= 100 && 
          (testval % 4) == 0
           true
       else
           false
       end
    end
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifelse_oror_and[queue, (1,)]
    for v = -1:101
        test_ocl(b, v)
        @fact cl.read(queue, b)[1] => comp(v)
    end
end

@clkernel test_ifelse_ororor(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if (testval % 2) == 0 || 
        testval <= 0 || 
        testval >= 100 || 
        testval == 10
        b[gid] = true
    else
        b[gid] = false
    end
    return
end 

facts("Test IfElse_OrOrOr") do
    comp = (testval) -> begin
         if (testval % 2) == 0 || 
            testval <= 0 || 
            testval >= 100 || 
            testval == 10
            true
        else
            false
        end
    end
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifelse_ororor[queue, (1,)]
    for v = -1:101
        test_ocl(b, v)
        @fact cl.read(queue, b)[1] => comp(v)
    end
end

@clkernel test_ifelse_and_or_and(b::Vector{Bool}, x::Int, y::Int) = begin
    gid = get_global_id(0)
    if x >= 0 && x < 10 || y >= 0 && y < 10
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfElse_And_Or_And") do
    comp = (x, y) -> begin
        if x >= 0 && x < 10 || y >= 0 && y < 10
            true
        else
            false
        end
    end 
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifelse_and_or_and[queue, (1,)]
    for x = -1:11, y = -1:11
        test_ocl(b, x, y)
        @fact cl.read(queue, b)[1] => comp(x, y)
    end
end

@clkernel test_ifelse_or_and_or(b::Vector{Bool}, x::Int, y::Int) = begin
    gid = get_global_id(0)
    if (x < 0 || x >= 10) && (y < 0 || y >= 10)
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfElse_Or_And_Or") do
    comp = (x, y) -> begin
        if (x < 0 || x >= 10) && (y < 0 || y >= 10)
            true
        else
            false
        end
    end 
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifelse_or_and_or[queue, (1,)]
    for x = -1:11, y = -1:11
        test_ocl(b, x, y)
        @fact cl.read(queue, b)[1] => comp(x, y)
    end
end

@clkernel test_ifor(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if testval >= 0 || testval < 100
        b[gid] = true
    else
        b[gid] = false
    end 
    return
end

facts("Test IfOr") do
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifor[queue, (1,)]
    test_ocl(b, -1)
    @fact cl.read(queue, b)[1] => true
    test_ocl(b, 1)
    @fact cl.read(queue, b)[1] => true
    test_ocl(b, 100)
    @fact cl.read(queue, b)[1] => true
end

@clkernel test_iforandor(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if ((testval % 2) ==0 || 
         testval <= 0 && 
         testval >= 100 || 
        (testval % 4) ==0)
        b[gid] = true
    else
        b[gid] = false
    end
    return 
end

facts("Test IfOrAndOr") do
    comp = (testval) -> begin
        if ((testval % 2) == 0 || 
             testval <= 0 && 
             testval >= 100 || 
            (testval % 4) == 0)
            true
        else
            false
        end
    end
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_iforandor[queue, (1,)]
    for v = -1:101
        test_ocl(b, v)
        @fact cl.read(queue, b)[1] => comp(v)
    end
end

@clkernel test_iforor(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if testval >= 0 || testval < 100 || testval == 20
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfOrOr") do
    comp = (testval) -> begin
        if testval >= 0 || testval < 100 || testval == 20
            true
        else
            false
        end
    end
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_iforor[queue, (1,)]
    for v = -1:101
        test_ocl(b, v)
        @fact cl.read(queue, b)[1] => comp(v)
    end
end

@clkernel test_ifororand(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if ((testval % 2) == 0 || 
        testval <= 0 || 
        testval >= 100 && 
        (testval % 4) == 0)
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfOrOrAnd") do
    comp = (testval) -> begin
     if ((testval % 2) == 0 || 
        testval <= 0 || 
        testval >= 100 && 
        (testval % 4) == 0)
            true
        else
            false
        end
    end
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifororand[queue, (1,)]
    for v = -1:101
        test_ocl(b, v)
        @fact cl.read(queue, b)[1] => comp(v)
    end
end

@clkernel test_ifororor(b::Vector{Bool}, testval::Int) = begin
    gid = get_global_id(0)
    if ((testval % 2) == 0 || testval <=0 || testval>=100 || testval==10)
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test IfOrOrOr") do
    comp = (testval) -> begin
        if ((testval % 2) == 0 || testval <=0 || testval>=100 || testval==10)
            true
        else
            false
        end
    end
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_ifororor[queue, (1,)]
    for v = -1:101
        test_ocl(b, v)
        @fact cl.read(queue, b)[1] => comp(v)
    end
end

@clkernel test_if_ifelseifelseelse_else(b::Vector{Bool}, x::Int, y::Int, z::Int) = begin
    gid = get_global_id(0)
    if x
        if y
            b[gid] = true
        elseif z
            b[gid] = true
        else
            b[gid] = false
        end
    else
        b[gid] = false
    end
    return 
end

facts("Test If_IfElseIfElseElse_Else") do
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_if_ifelseifelseelse_else[queue, (1,)]
    test_ocl(b, 0, 0, 0)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 0, 1, 1)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 1, 1, 1)
    @fact cl.read(queue, b)[1] => true
    test_ocl(b, 1, 0, 1)
    @fact cl.read(queue, b)[1] => true
    test_ocl(b, 1, 0, 0)
    @fact cl.read(queue, b)[1] => false
end

@clkernel test_if_ifelse_else(b::Vector{Bool}, x::Int, y::Int) = begin
    gid = get_global_id(0)
    if x
        if y
            b[gid] = true
        else
            b[gid] = false
        end
    else
        b[gid] = false
    end
    return
end

facts("Test If_IfElse_Else") do
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_if_ifelse_else[queue, (1,)]
    test_ocl(b, 0, 0)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 0, 1)
    @fact cl.read(queue, b)[1] => false
    test_ocl(b, 1, 1)
    @fact cl.read(queue, b)[1] => true
    test_ocl(b, 1, 0)
    @fact cl.read(queue, b)[1] => false
end

@clkernel test_if_ifelse_else_ifelse(b::Vector{Int}, w::Int, x::Int, y::Int, z::Int) = begin
    gid = get_global_id(0)
    if w
        if x
            if y
                b[gid] = 1
            else
                b[gid] = 2
            end
        else
            if z
                b[gid] = 3
            else
                b[gid] = 4
            end
        end
    else
        b[gid] = -1
    end
    return 
end

facts("Test If_IfElse_else_IfElse") do
    b = cl.Buffer(Int, ctx, 1)
    test_ocl = test_if_ifelse_else_ifelse[queue, (1,)]
    test_ocl(b, 0, 0, 0, 0)
    @fact cl.read(queue, b)[1] => -1
    test_ocl(b, 1, 0, 0, 0)
    @fact (cl.read(queue, b)[1]) => 4
    test_ocl(b, 1, 0, 0, 1)
    @fact (cl.read(queue, b)[1]) => 3
    test_ocl(b, 1, 1, 0, 0)
    @fact (cl.read(queue, b)[1]) => 2
    test_ocl(b, 1, 1, 1, 0)
    @fact (cl.read(queue, b)[1]) => 1
end 

#TODO: this doesn't compile as x && y is a Union(Int, Bool) 
@clkernel test_if_if_else(b::Vector{Bool}, x::Int, y::Int) = begin
    gid = get_global_id(0)
    if x && y > 0
        b[gid] = true
    else
        b[gid] = false
    end
    return
end

facts("Test If_IfElse") do
    b = cl.Buffer(Bool, ctx, 1)
    test_ocl = test_if_if_else[queue, (1,)]
    test_ocl(b, 0, 0)
    @fact (cl.read(queue, b)[1]) => false
    test_ocl(b, 0, 1)
    @fact (cl.read(queue, b)[1]) => false
    test_ocl(b, 1, 1)
    @fact (cl.read(queue, b)[1]) => true
end
