using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.CLAst

using OpenCL.SourceGen
using OpenCL.Compiler

const rmline = OpenCL.Compiler.rm_linenum!

#---- test functions ---
function test1(x)
    return x += 1
end

function test2(x)
    y = float32(x) + float32(2)
    return y ^ 10
end

function test3(x, y)
    z = x + y
    return 
end

function test4(x::Array{Float64, 1}, y)
    gid = 1
    x[gid] = y
    return x
end

function test5(x)
    if x > 10
        x = x < 5 ? 1 : 2
    else
        x += 2
    end
    return x
end

function get_global_id(x)
    x + 2
    y + 3
    return uint32(x)::Uint32
end

function get_global_size(x)
    x + 2
    y + 3
    return uint32(x)::Uint32
end

device = cl.devices()[end-1]
ctx = cl.Context(device)
queue = cl.CmdQueue(ctx)
#device, ctx, queue = cl.create_compute_context()
@show device[:platform]

uncompressed_ast(l::LambdaStaticData) = begin
    if isa(l.ast,Expr)
        return l.ast
    else
        return ccall(:jl_uncompress_ast, Any, (Any,Any), l, l.ast) 
    end
end

macro clkernel(func)
    f, n = gensym("func"), gensym("n")
    
    orig_name = func.args[1].args[1]
    func.args[1].args[1] = symbol(f)
    
    quote
        local func = eval($(esc(func)))
        # lookup method name from method table
        local name = func.env.name
        if length(func.env) != 1
            error("more than one kernel with name $name")
        end
        # lookup method signature from first method in method table
        local typs = func.env.defs.sig
        for ty in typs
            if !isleaftype(ty)
                error("function signature nonleaftype $ty")
            end
        end
        println($("$orig_name compile time:"))
        @time begin
        local exprs = code_typed(func, typs)
        if length(exprs) == 0
            error("function could not be compiled for attribute types:: $typs")
        end
        if length(exprs) > 1
            error("more than one typed ast produced!")
        end
        local expr = first(exprs)
        #println(expr)

        kern_ctx, kernel = build_kernel($("$orig_name"), expr)
        local io  = IOBuffer()
        print(io, "#pragma OPENCL EXTENSION cl_amd_printf : enable\n")
        print(io, "typedef struct Range {long start; long step; long len;} Range;\n")
        for n in unique(keys(kern_ctx.funcs))
            clsource(io, kern_ctx.funcs[n])
            println(io)
        end
        clsource(io, kernel)
        local src = bytestring(io.data)
        #println(src)
        
        # TODO: return a fucntion that takes a context
        # build the source and store in global cache
        local p = cl.Program(ctx, source=src) |> cl.build!
        global $(orig_name) = cl.Kernel(p, $("$orig_name"))
    end
    end
end

macro cljit(func)
    @clkernel(func)
end

function juliaref{T}(a::Vector{T},
                     b::Vector{T},
                     c::Vector{T},
                     count::Cuint)
   for gid in 1:count
       c[gid] = exp(a[gid]) + log(b[gid])
   end
   return
end

function testadd3(a)
    a * 2.0
    return a * 2.0
end

function testadd2(a, b)
    test1 = a
    test2 = b
    a * b
    return testadd3(a) + testadd3(b)
end

function testadd(a, b)
    test1 = exp(a)
    test2 = log(b)
    return testadd2(test1, test2)
end

@assert isa(eval(:testadd), Function)

@clkernel test6(a::Vector{Float64}, 
                b::Vector{Float64}, 
                c::Vector{Float64},
                count::Cuint) = begin
    gid = get_global_id(0)
    if gid < count
        c[gid] = exp(a[gid]) + log(b[gid])
    end
    return
end


const test7 = """
typedef struct Range {
   long start;
   long step;
   long len;
} Range;

__kernel void testcl(__global double *a,
                     __global double *b, 
                     __global double *c,
                     unsigned int count)
{

  size_t gid = get_global_id(0);
  if (gid < count) {
      c[gid] = exp(a[gid]) + log(b[gid]);
  }
  return;
}
"""

@assert isa(test6, cl.Kernel)

#--------------------------

function can_compile(src)
    try
        ctx = cl.create_some_context()
        p = cl.Program(ctx, source=src) |> cl.build!
        return true
    catch err
        return false
    end
end

#facts("Builtins") do
#    for ty in (:Int8, :Uint8, :Int16, :Uint16, :Int32, :Uint32) #:Int64, :Uint64)
#        @eval begin
#            expr = first(code_typed(test1, ($ty,)))
#            expr = expr.args[end].args[2].args[2]
#            ast1 = visit(expr)
#            code1 = clsource(ast1)
#            ast2 = CBinOp(CTypeCast(CName("x", $ty), Int64),
#                          CAdd(),
#                          CNum(1, Int64),
#                          Int64)
#            @fact ast1 => ast2
#            code2 = clsource(ast2) 
#            @fact code1 => code2
#        end
#    end
#    
#    expr = first(code_typed(test1, (Int64,)))
#    expr = expr.args[end].args[2].args[2]
#    ast1 = visit(expr)
#    code1 = clsource(ast1)
#    ast2 = CBinOp(CName("x", Int64),
#                  CAdd(),
#                  CNum(1, Int64),
#                  Int64)
#    @fact ast1 => ast2
#    code2 = clsource(ast2) 
#    @fact code1 => code2
#
#    expr = first(code_typed(test1, (Uint64,)))
#    expr = expr.args[end].args[2].args[2]
#    ast1 = visit(expr)
#    code1 = clsource(ast1)
#    ast2 = CBinOp(CName("x", Uint64),
#                  CAdd(),
#                  CNum(1, Uint64),
#                  Uint64)
#    @fact ast1 => ast2
#    code2 = clsource(ast2) 
#    @fact code1 => code2
#
#    for ty in (:Float32, :Float64)
#        @eval begin 
#            expr = first(code_typed(test1, ($ty,)))
#            expr = expr.args[end].args[2].args[2]
#            ast1 = visit(expr)
#            code1 = clsource(ast1) 
#            ast2 = CBinOp(CName("x", $ty),
#                          CAdd(),
#                          CNum(1, $ty),
#                          $ty)
#            @fact ast1 => ast2
#            code2 = clsource(ast2)
#            @fact code1 => code2
#        end
#    end
#
#    # cast floating point values
#    top_expr = first(code_typed(test2, (Float64,)))
#    expr = top_expr.args[end].args[2].args[2]
#    @fact visit(expr) => CBinOp(CTypeCast(CName("x", Float64), Float32),
#                                CAdd(),
#                                CNum(2.0, Float32),
#                                Float32)
#    @fact clsource(visit(expr)) => "((float) x) + 2.0f"
#
#    # compile block ast nodes
#    expr = top_expr.args[end]
#    @fact clsource(visit(expr)) => "{{\n  y = ((float) x) + 2.0f;\n  return(pown(y, 10));\n}}\n"
#
#    # compile lambda static functions
#    expr = top_expr 
#    #@show clsource(visit(expr))
#
#    expr = first(code_typed(test3, (Float32, Float32)))
#    #@show clsource(visit(expr))
#    expr = first(code_typed(test4, (Array{Float64,1},Float32)))
#    #println(clsource(visit(expr)))

#    a = rand(Float64,  500_000)
#    b = rand(Float64,  500_000)
#    c = zeros(Float64, 500_000)

#    a_buff = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf=a)
#    b_buff = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf=b)
#    c_buff = cl.Buffer(Float64, ctx, :rw, length(a))

#    println("TEST Julia")

#    for _ = 1:1
#        @time juliaref(a, b, c, uint32(length(a)))
#    end

#    p = cl.Program(ctx, source=test7) |> cl.build!
#    t7 = cl.Kernel(p, "testcl")

#    for i = 1:2
#        tic()
#        cl.call(queue, t7, size(a), nothing,
#                a_buff, b_buff, c_buff, uint32(length(a))) 
#        r = cl.read(queue, c_buff)
#        toc()
#    end

#    println("TEST 6")
#    local r::Vector{Float32}
#    for i = 1:2
#        tic()
##        cl.call(queue, test6, size(a), nothing,
#                a_buff, b_buff, c_buff, int32(length(a))) 
##        r = cl.read(queue, c_buff)
#        toc()
#    end
#    @show norm(r - (exp(a) + log(b)))
#    #@fact isapprox(norm(r - (exp(a) + log(b))), zero(Float32)) => #true
#end

function compile_anonfunc(f, types)
    if isgeneric(f) || (isdefined(f, :env) && isa(f.env, Symbol))
        error("not an anonymous function")
    end
    (tree, ty) = Base.typeinf(f.code, types,())
    ast = ccall(:jl_uncompress_ast, Any, (Any, Any), f.code, tree)
    return (ast, ty)
end
    
#f = (x) -> x + 2
#@show compile_anonfunc(f, (Int32,)) 

function twiddle(u::Uint32, v::Uint32)
    t1 = ((u & 0x80000000) | (v & 0x7FFFFFFF)) >> int32(1)
    local t2::Uint32
    if v & uint32(1) == zero(Uint32)
        t2 = uint32(0x0)
    else
        t2 = uint32(0x9908B0DF)
    end
    return t1 $ t2
end

@clkernel generate_state(state::Vector{Uint32}) = begin
    n = uint32(624)
    m = uint32(397)
    for i = int32(0:(n - m - 1))
        state[i] = state[i] + m
    end
    for i = int32(0:(n - m - 1))
        state[i] = state[i+m] $ twiddle(state[i], state[i+1])
    end
    for i = int32((n - m):(n - 2))
        state[i] = state[i+m-n] $ twiddle(state[i], state[i+1])
    end
    state[n-1] = state[m-1] $ twiddle(state[n-1], state[0])
    return
end

function generate_state_julia(state::Vector{Uint32})
    n = uint32(624)
    m = uint32(397)
    for i = int32(0:(n - m - 1))
        state[i] = state[i] + m
    end
    for i = int32(0:(n - m - 1))
       state[i] = state[i+m] $ twiddle(state[i], state[i+1])
    end
    for i = int32((n - m):(n - 2))
        state[i] = state[i+m-n] $ twiddle(state[i], state[i+1])
    end
    state[n-1] = state[m-1] $ twiddle(state[n-1], state[0])
    return
end

@clkernel seed(s::Uint32, state::Vector{Uint32}) = begin
    n = uint32(624)
    m = uint32(397)
    state[0] = s & 0xFFFFFFFF
    for i = int32(1:(n-1))
        state[i] = 1812433253 * (state[i-1] $ (state[i-1] >> int32(30))) + uint32(i)
        state[i] = state[i] & 0xFFFFFFFF
    end
    return generate_state2(state)
end

@assert isa(seed, cl.Kernel)

function random_number(state::Vector{Uint32}, p::Cuint)
    x = state[p]
    x $= (x >>> int32(11))
    x $= (x <<< int32(7)) & 0x9D2C5680
    x $= (x <<< int32(15)) & 0xEFC60000
    return x $ (x >>> int32(8))
end
    
@clkernel fill(state::Vector{Uint32},
               vector::Vector{Uint32},
               offset::Cuint) = begin
    i = get_global_id(0)
    vector[offset + i] = random_number(state, i)
    return
end

@assert isa(fill, cl.Kernel)


#src = open(readall, "test.cl")
#prg = cl.Program(ctx, source=src) |> cl.build!
#generate_state = cl.Kernel(prg, "generate_state")
#seed = cl.Kernel(prg, "seed")
#fill = cl.Kernel(prg, "fill")

function seed_mersenne!{T}(state_buffer::cl.Buffer{T})
    n = length(state_buffer)
    cl.call(queue, seed, n, nothing, uint32(n), state_buffer)
    return
end

#seed_mersenne(Float32)

function test_fill{T}(b::Vector{T})
    n = 624
    m = 397 
    len = length(b)

    buffer = cl.Buffer(T, ctx, (:rw, :copy), hostbuf=b)
    
    state_buffer = cl.Buffer(T, ctx, :rw, n)
    seed_mersenne!(state_buffer)
    
    cl.set_arg!(fill, 1, state_buffer)
    cl.set_arg!(fill, 2, buffer)
    
    p = 0
    while true
        cnt = 0
        if len - p >= n
            cnt = n
        else
            cnt = len - p
        end
        cl.set_arg!(fill, 3, uint32(p))
        cl.enqueue_kernel(queue, fill, (cnt,))
        p += n
        if p >= len
            break
        end
        cl.set_arg!(generate_state, 1, state_buffer)
        cl.enqueue_kernel(queue, generate_state, (1,), (1,))
    end
    return cl.read(queue, buffer)
end

#z = zeros(Float32, 1_000_000)
#@time rand(Float32, 1_000_000)
#for _ = 1:10
#    @time test_fill(z)
#end

#@show test_fill(z)[1:50]

@clkernel generate_sin(a::Vector{Float32}, 
                       b::Vector{Float32}) = begin
    gid = get_global_id(0)
    n   = get_global_size(0)
    
    r = float32(gid) / float32(n)

    # sin wave with 8 oscillations
    y = r * (16.0f0 * 2.1415f0)

    # x is a range from -1 to 1
    a[gid] = r * 2.0f0 - 1.0f0

    # y is a sin wave
    b[gid] = sin(y)

    return
end

function generate_sin_julia(a::Vector{Float32}, b::Vector{Float32})
    n = length(a)
    for gid in 1:n
        r = float32(gid) / float32(n)
        # sin wave with 8 oscillations
        y = r * (16.0f0 * 2.1415f0)
        # x is a range from -1 to 1
        a[gid] = r * 2.0f0 - 1.0f0
        # y is a sin wave
        b[gid] = sin(y)
    end
    return deepcopy(b)
end
       
#n = 1_000_000
#a = cl.Buffer(Float32, ctx, n)
#b = cl.Buffer(Float32, ctx, n)

#info("OpenCL")
#@time begin
#for _ in 1:1
#        evt = cl.call(queue, generate_sin, (n,), nothing, a, b)
#        r = cl.read(queue, b)
#    end
#end

#info("JUlia")
#a = Array(Float32, n)
#b = Array(Float32, n)
#@time begin
#for _ in 1:1
#    comp_func(a, b)
#end
#end


@cljit juliat(r::Vector{Float32}, 
              i::Vector{Float32},
              output::Vector{Uint16},
              maxiter::Int32,
              len::Int32) = begin
    gid = get_global_id(0)
    if gid < len 
        nreal = 0.0f0
        real = r[gid]
        imag = i[gid]
        output[gid] = uint16(0)
        for curiter = 1:maxiter
            tmp = abs(real*real + imag*imag)
            if (tmp > 4.0f0) && !(isnan(tmp))
                output[gid] = uint16(curiter - 1)
            end
            nreal = real*real - imag*imag - 0.5f0
            imag = 2.0f0*real*imag + 0.75f0
            real = nreal
        end
    end
    return
end

test_julia(r::Vector{Float32}, 
           i::Vector{Float32},
           output::Array{Uint16, 2},
           maxiter::Int32) = begin
    for gid = 1:length(output) 
        nreal = 0.0f0
        real = r[gid]
        imag = i[gid]
        output[gid] = 0
        
        for curiter = 1:maxiter
            tmp = real * real + imag * imag 
            if (tmp > 4.0) && !(isnan(tmp))
                output[gid] = uint16(curiter - 1)
            end
            nreal = real*real - imag*imag - 0.5f0
            imag = 2.0f0 * real * imag + 0.75f0
            real = nreal
        end
    end
end


@show isa(juliat, cl.Kernel)


function julia_opencl(q::Array{Complex64}, maxiter::Int64)
    r = [real(i) for i in q]
    i = [imag(i) for i in q]
   
    r_buff = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=r)
    i_buff = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=i)
    o_buff = cl.Buffer(Uint16,  ctx, :rw, length(q))
    
    cl.call(queue, juliat, length(q), nothing, r_buff, i_buff, o_buff, int32(maxiter), int32(length(q)))
    #cl.copy!(queue, out, o_buff)
    out = cl.read(queue, o_buff)
    return reshape(out, size(q))
end

#using PyPlot
#w = 2048 * 3;
#h = 2048 * 3;
#q = [complex64(r,i) for i=1:-(2.0/w):-1, r=-1.5:(3.0/h):1.5];
#m = nothing
#for _= 1:5
#    @time m = julia_opencl(q, 200);
#end

#m = Array(Uint16, size(q))
#r = Float32[real(i) for i in q]
#i = Float32[imag(i) for i in q]
#for _ = 1:3
#    @time test_julia(r, i, m, int32(200))
#end
#imshow(m, cmap="RdGy", extent=[-1.5,1.5,-1,1]);


const RISKFREE = 0.02
const VOLATILITY = 0.30

function cnd(d)
    #TODO: global variables
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    K = 1.0 / (1.0 + 0.2316419 * abs(d))
    ret_val = (RSQRT2PI * exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0.0
        ret_val = 1.0 - ret_val
    end
    return ret_val
end

@cljit black_scholes(call_result::Vector{Cdouble},
                     put_result::Vector{Cdouble},
                     S::Vector{Cdouble},
                     X::Vector{Cdouble},
                     T::Vector{Cdouble},
                     R::Cdouble,
                     V::Cdouble,
                     len::Int) = begin
    i = get_global_id(0)
    if i >= len
        return
    end
    sqrtT = sqrt(T[i])
    d1 = (log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd(d1)
    cndd2 = cnd(d2)
    expRT = exp((-1. * R) * T[i])
    call_result[i] = S[i] * cndd1 - X[i] * expRT * cndd2
    put_result[i]  = X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1)
    return
end

# TODO: if setindex! is called when array variable is misnamed
# check global scope 

black_scholes_julia(call_result::Vector{Cdouble},
                    put_result::Vector{Cdouble},
                    S::Vector{Cdouble},
                    X::Vector{Cdouble},
                    T::Vector{Cdouble},
                    R::Cdouble,
                    V::Cdouble,
                    len::Int) = begin
    for i = 1:len
        sqrtT = sqrt(T[i])
        d1 = (log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
        d2 = d1 - V * sqrtT
        cndd1 = cnd(d1)
        cndd2 = cnd(d2)
        expRT = exp((-1. * R) * T[i])
        call_result[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
        put_result[i]  = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))
    end
    return
end

function randfloat(rand_var, low, high)
    return (1.0 - rand_var) * low + rand_var * high
end

function test_sholes()
    OPT_N = 100_000
    iterations = 1
    
    stockPrice   = randfloat(rand(Cdouble, OPT_N), 5.0, 30.0)
    optionStrike = randfloat(rand(Cdouble, OPT_N), 1.0, 100.0)
    optionYears  = randfloat(rand(Cdouble, OPT_N), 0.25, 10.0)

    callResultJulia = zeros(Cdouble, OPT_N)
    putResultJulia  = -ones(Cdouble, OPT_N)

    tic()
    for i in 1:iterations
        black_scholes_julia(callResultJulia, putResultJulia, stockPrice,
                            optionStrike, optionYears, RISKFREE, VOLATILITY, OPT_N)
    end
    t = toc()
    info("Julia Time: $((1000 * t) / iterations) msec per iteration")
    
    callResultOpenCL = zeros(Cdouble, OPT_N)
    putResultOpenCL  = -ones(Cdouble, OPT_N)
    
    d_callResult   = cl.Buffer(Cdouble, ctx, :rw, length(callResultOpenCL))
    d_putResult    = cl.Buffer(Cdouble, ctx, :rw, length(putResultOpenCL))
    d_stockPrice   = cl.Buffer(Cdouble, ctx, (:r, :copy), hostbuf=stockPrice)
    d_optionStrike = cl.Buffer(Cdouble, ctx, (:r, :copy), hostbuf=optionStrike)
    d_optionYears  = cl.Buffer(Cdouble, ctx, (:r, :copy), hostbuf=optionYears)
   
    # create a kernel function with queue, global size OPT_N
    black_sholes_ocl = black_sholes[queue, (OPT_N,)]

    tic()
    for i = 1:iterations
        black_sholes_ocl(d_callResult, d_putResult, d_stockPrice, 
                         d_optionStrike, d_optionYears, RISKFREE, VOLATILITY, OPT_N)
        cl.enqueue_barrier(queue)
    end
    cl.copy!(queue, callResultOpenCL, d_callResult)
    cl.copy!(queue, putResultOpenCL,  d_putResult)
    t = toc()
    
    info("OpenCL Time: $((1000 * t) / iterations) msec per iteration")
    
    delta  = abs(callResultJulia - callResultOpenCL)
    L1norm = sum(delta) / sum(abs(callResultOpenCL))
    
    info("L1 norm (OpenCL): $L1norm")
    info("Max absolute error: $(maximum(delta))")
end

test_sholes()
