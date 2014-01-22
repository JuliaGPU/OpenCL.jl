using FactCheck 

import OpenCL.CLAst 
const clast = OpenCL.CLAst

import OpenCL.CLSourceGen
const clprint = OpenCL.CLSourceGen.clprint

facts("Generation") do
    ast = clast.CBinOp(clast.CNum(1), 
                       clast.CAdd(), 
                       clast.CNum(1),
                       Int64)
    code = sprint() do io
        clprint(io, ast, 0)
    end 
    @fact code => "(1 + 1)"

    ast = clast.CUnaryOp(clast.CUAdd(), 
                         clast.CName("foo"), 
                         Int64)
    code = sprint() do io
        clprint(io, ast, 0)
    end 
    @fact code => "(++(foo))"

    ast = clast.CAssign(clast.CName("foo"), 
                        clast.CBinOp(clast.CNum(1),
                                     clast.CAdd(),
                                     clast.CNum(1),
                                     Int64),
                       Int64)
    code = sprint() do io
        clprint(io, ast, 0)
    end 
    @fact code => "foo = (1 + 1)"

    ast = clast.CBlock([clast.CFunctionCall("foo"), 
                        clast.CFunctionCall("bar")])
    code = sprint() do io
        clprint(io, ast, 0)
    end 
    @fact code => "{{\n\tfoo();\n\tbar();\n}}\n"

    ast = clast.CFor(clast.CAssign(clast.CName("i"), 
                                   clast.CNum(0),
                                   Int64),
                     clast.CBinOp(clast.CName("i"), 
                                  clast.CLtE(), 
                                  clast.CNum(10),
                                  Int64),
                     clast.CUnaryOp(clast.CUAdd(),
                                    clast.CName("i"),
                                    Int64),
                     clast.CBlock([clast.CAssign(clast.CName("i"),
                                                 clast.CNum(1),
                                                 Int64)]))
    code = sprint() do io
        clprint(io, ast, 0)
    end
    @fact code => "for (i = 0; (i <= 10); (++(i))) {{\n\ti = 1;\n}}\n"


    ast = clast.CAssign(clast.CSubscript(clast.CName("test"),
                                         clast.CIndex(clast.CNum(1))),
                        clast.CNum(10),
                        Int64)
    code = sprint() do io
        clprint(io, ast, 0)
    end
    @fact code => "test[1] = 10"
end
