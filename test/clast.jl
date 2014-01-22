using FactCheck 

import OpenCL.CLAst 
const clast = OpenCL.CLAst

import OpenCL.CLSourceGen
const clprint = OpenCL.CLSourceGen.clprint

facts("Generation") do
    ast = clast.CBinOp(clast.CNum{Int64}(1), clast.CAdd(), 
                       clast.CNum{Int64}(1))
    code = sprint() do io
        clprint(io, ast, 0)
    end 
    @fact code => "(1 + 1)"

    ast = clast.CUnaryOp(clast.CUAdd(), clast.CName("foo"))
    code = sprint() do io
        clprint(io, ast, 0)
    end 
    @fact code => "(++(foo))"

    ast = clast.CAssign(clast.CName("foo"), 
                        clast.CBinOp(clast.CNum{Int64}(1), clast.CAdd(),
                                     clast.CNum{Int64}(1)))
    code = sprint() do io
        clprint(io, ast, 0)
    end 
    @fact code => "foo = (1 + 1)"

    ast = clast.CBlock([clast.CFunctionCall("foo"), 
                        clast.CFunctionCall("bar")])
    code = sprint() do io
        clprint(io, ast, 0)
    end 
    @fact code => "{{\n\tfoo();\n\tbar();\n}}"
end
