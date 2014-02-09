using FactCheck 

using OpenCL.CLAst 
using OpenCL.SourceGen
using OpenCL.Compiler

#TODO: Ensure correct number printing
facts("OpenCL source generation") do
    ast = CBinOp(CNum(1), CAdd(), CNum(1), Int64)
    code = clsource(ast) 
    @fact code => "(1l) + (1l)"

    ast = CUnaryOp(CUAdd(), CName("foo"), Int64)
    code = clsource(ast) 
    @fact code => "(++(foo))"

    ast = CAssign(CName("foo"), 
                  CBinOp(CNum(1), CAdd(), CNum(1), Int64),
                  Int64)
    code = clsource(ast)
    @fact code => "foo = (1l) + (1l)"

    ast = CBlock([CFunctionCall("foo", [], Void), 
                  CFunctionCall("bar", [], Void)])
    code = clsource(ast) 
    @fact code => "{{\n\tfoo();\n\tbar();\n}}\n"

    ast = CFor(CAssign(CName("i"), 
                       CNum(0),
                       Int64),
                     CBinOp(CName("i"), 
                            CLtE(), 
                            CNum(10),
                            Int64),
                     CUnaryOp(CUAdd(),
                              CName("i"),
                              Int64),
                     CBlock([CAssign(CName("i"),
                                     CNum(1),
                                     Int64)]))
    code = clsource(ast) 
    @fact code => "for (i = 0l; (i) <= (10l); (++(i))) {{\n\ti = 1l;\n}}\n"

    ast = CAssign(CSubscript(CName("test"),
                             CIndex(CNum(1)),
                             Int),
                  CNum(10),
                  Int64)
    code = clsource(ast) 
    @fact code => "test[1l] = 10l"

    #TODO: kernel.. 
end
