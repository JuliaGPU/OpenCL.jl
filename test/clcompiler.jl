using FactCheck 

using OpenCL.CLAst

using OpenCL.CLSourceGen

import OpenCL.CLCompiler
const visit = OpenCL.CLCompiler.visit

function test1(x)
    return x += 1
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
                  CTypeCast(CNum(1), Uint64),
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
end
