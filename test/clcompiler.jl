using FactCheck 

using OpenCL.CLAst

using OpenCL.CLSourceGen

import OpenCL.CLCompiler
const visit = OpenCL.CLCompiler.visit

function test1(x)
    return x += 1
end

facts("Builtins") do
    expr = first(code_typed(test1, (Int64,)))
    expr = expr.args[end].args[2].args[2]
    ast1 = visit(expr)
    code1 = clsource(ast1)
    ast2 = CBinOp(CName("x", Int64),
                  CAdd(),
                  CNum(1),
                  Int64)
    @fact ast1 => ast2
    code2 = clsource(ast2) 
    @fact code1 => code2
end



