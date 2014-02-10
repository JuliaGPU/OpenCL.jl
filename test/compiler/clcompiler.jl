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
