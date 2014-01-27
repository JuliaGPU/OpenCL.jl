using FactCheck 

using OpenCL.CLAst 
using OpenCL.CLSourceGen
using OpenCL.CLCompiler2

facts("Generation") do
    ast = CBinOp(CNum(1), 
                 CAdd(), 
                 CNum(1),
                 Int64)
    code = clsource(ast) 
    @fact code => "1 + 1"

    ast = CUnaryOp(CUAdd(), 
                   CName("foo"), 
                   Int64)
    code = clsource(ast) 
    @fact code => "++(foo)"

    ast = CAssign(CName("foo"), 
                  CBinOp(CNum(1),
                         CAdd(),
                         CNum(1),
                         Int64),
                  Int64)
    code = clsource(ast)
    @fact code => "foo = 1 + 1"

    ast = CBlock([CFunctionCall("foo", [], Void), 
                  CFunctionCall("bar", [], Void)])
    code = clsource(ast) 
    @fact code => "{{\n  foo();\n  bar();\n}}\n"

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
    @fact code => "for (i = 0; i <= 10; ++(i)) {{\n  i = 1;\n}}\n"

    ast = CAssign(CSubscript(CName("test"),
                             CIndex(CNum(1)),
                             Int),
                  CNum(10),
                  Int64)
    code = clsource(ast) 
    @fact code => "test[1] = 10"
end

facts("Parse Expr") do
    expr = :(i * 1)
    @fact clsource(visit(expr)) => "i * 1"

    expr = :(i / 1)
    @fact clsource(visit(expr)) => "i / 1"
    
    expr = :(i + 1)
    @fact clsource(visit(expr)) => "i + 1"
    
    expr = :(i - 1)
    @fact clsource(visit(expr)) => "i - 1"
    
    expr = :(i % 1)
    @fact clsource(visit(expr)) => "i % 1"

    expr = :(i < 1) 
    @fact clsource(visit(expr)) => "i < 1"
    
    expr = :(i <= 1) 
    @fact clsource(visit(expr)) => "i <= 1"
    
    expr = :(i > 1) 
    @fact clsource(visit(expr)) => "i > 1"
    
    expr = :(i >= 1) 
    @fact clsource(visit(expr)) => "i >= 1"

    expr = :(i == 1)
    @fact clsource(visit(expr)) => "i == 1"

    expr = :(i != 1)
    @fact clsource(visit(expr)) => "i != 1"

    expr = :(i || 1)
    @fact clsource(visit(expr)) => "i || 1"

    expr = :(i && 1)
    @fact clsource(visit(expr)) => "i && 1"
    
    expr = :(!(i))
    @fact clsource(visit(expr)) => "!(i)"

    expr = :(i += 1)
    @fact clsource(visit(expr)) => "i = i + 1"

    expr = :(i -= 1)
    @fact clsource(visit(expr)) => "i = i - 1"
    
    expr = :(i *= 1)
    @fact clsource(visit(expr)) => "i = i * 1"
    
    expr = :(i /= 1)
    @fact clsource(visit(expr)) => "i = i / 1"

    expr = :(i << 1)
    @fact clsource(visit(expr)) => "i << 1"
    
    expr = :(i >> 1)
    @fact clsource(visit(expr)) => "i >> 1"
    
    expr = :(for i in 0:10; end)
    @fact clsource(visit(expr)) => "for (int i = 0; i <= 10; i = i + 1) {{\n}}\n" 
    
    expr = :(for i in 0:2:10; end)
    @fact clsource(visit(expr)) => "for (int i = 0; i <= 10; i = i + 2) {{\n}}\n" 

    expr = :(while i < 10; end)
    @fact clsource(visit(expr)) => "while (i < 10) {{\n}}\n"

    expr = :(if i == 1; i += 2; end)
    @fact clsource(visit(expr)) => "if (i == 1) {{\n  i = i + 2;\n}}\n"
    
    expr = :(if i == 1; i += 2; else; i += 3; end)
    @fact clsource(visit(expr)) => 
        "if (i == 1) {{\n  i = i + 2;\n}}\nelse {{\n  i = i + 3;\n}}\n"
    
    expr = :(if i == 1 
               i += 2 
             elseif i == 2 
                 if i == 2
                    i == 4
                 end
               i += 3; 
             else
               i += 4
             end)
    @fact clsource(visit(expr)) =>
        "if (i == 1) {{\n  i = i + 2;\n}}\nelse {{\n  if (i == 2) {{\n    if (i == 2) {{\n      i == 4;\n    }}\n    i = i + 3;\n  }}\n  else {{\n    i = i + 4;\n  }}\n}}\n"

end
