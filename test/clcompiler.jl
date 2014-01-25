using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.CLAst

using OpenCL.CLSourceGen
import OpenCL.CLCompiler

const visit = OpenCL.CLCompiler.visit
const rmline = OpenCL.CLCompiler.rm_linenum!

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
    return uint32(1)
end

function test6(a::Vector{Float32}, 
               b::Vector{Float32}, 
               c::Vector{Float64}, 
               count::Cuint)
    gid = get_global_id(0)
    if gid < count
        c[gid] = a[gid] + b[gid]
    end
    return
end

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

facts("Builtins") do
    for ty in (:Int8, :Uint8, :Int16, :Uint16, :Int32, :Uint32) #:Int64, :Uint64)
        @eval begin
            expr = first(code_typed(test1, ($ty,)))
            expr = expr.args[end].args[2].args[2]
            ast1 = visit(expr)
            code1 = clsource(ast1)
            ast2 = CBinOp(CTypeCast(CName("x", $ty), Int64),
                          CAdd(),
                          CNum(1, Int64),
                          Int64)
            @fact ast1 => ast2
            code2 = clsource(ast2) 
            @fact code1 => code2
        end
    end
    
    expr = first(code_typed(test1, (Int64,)))
    expr = expr.args[end].args[2].args[2]
    ast1 = visit(expr)
    code1 = clsource(ast1)
    ast2 = CBinOp(CName("x", Int64),
                  CAdd(),
                  CNum(1, Int64),
                  Int64)
    @fact ast1 => ast2
    code2 = clsource(ast2) 
    @fact code1 => code2

    expr = first(code_typed(test1, (Uint64,)))
    expr = expr.args[end].args[2].args[2]
    ast1 = visit(expr)
    code1 = clsource(ast1)
    ast2 = CBinOp(CName("x", Uint64),
                  CAdd(),
                  CNum(1, Uint64),
                  Uint64)
    @fact ast1 => ast2
    code2 = clsource(ast2) 
    @fact code1 => code2

    for ty in (:Float32, :Float64)
        @eval begin 
            expr = first(code_typed(test1, ($ty,)))
            expr = expr.args[end].args[2].args[2]
            ast1 = visit(expr)
            code1 = clsource(ast1) 
            ast2 = CBinOp(CName("x", $ty),
                          CAdd(),
                          CNum(1, $ty),
                          $ty)
            @fact ast1 => ast2
            code2 = clsource(ast2)
            @fact code1 => code2
        end
    end

    # cast floating point values
    top_expr = first(code_typed(test2, (Float64,)))
    expr = top_expr.args[end].args[2].args[2]
    @fact visit(expr) => CBinOp(CTypeCast(CName("x", Float64), Float32),
                                CAdd(),
                                CNum(2.0, Float32),
                                Float32)
    @fact clsource(visit(expr)) => "((float) x) + 2.0f"

    # compile block ast nodes
    expr = top_expr.args[end]
    @fact clsource(visit(expr)) => "{{\n\ty = ((float) x) + 2.0f;\n\treturn(pow(y, 10.0f));\n}}\n"

    # compile lambda static functions
    expr = top_expr 
    #@show clsource(visit(expr))

    expr = first(code_typed(test3, (Float32, Float32)))
    #@show clsource(visit(expr))
    expr = first(code_typed(test4, (Array{Float64,1},Float32)))
    println(clsource(visit(expr)))

    expr = first(code_typed(test6, (Array{Float32},
                                    Array{Float32},
                                    Array{Float64},
                                    Cuint)))
    @show rmline(expr)
    src = clsource(visit(expr))
    println(src)
    @fact can_compile(src) => true
end
