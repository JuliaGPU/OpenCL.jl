module CLSourceGen

import ..CLAst 

# see base/show.jl 292 for julia ast printing

Base.show(io::IO, node::CLAst.CMult) = print(io,"*")
Base.show(io::IO, node::CLAst.CAdd)  = print(io, "+")
Base.show(io::IO, node::CLAst.CUAdd) = print(io, "++")
Base.show(io::IO, node::CLAst.CSub)  = print(io, "-")
Base.show(io::IO, node::CLAst.CUSub) = print(io, "--")
Base.show(io::IO, node::CLAst.CDiv)  = print(io, "/")
Base.show(io::IO, node::CLAst.CMod)  = print(io, "%")
Base.show(io::IO, node::CLAst.CNot)  = print(io, "!")
#TODO: bitwise

Base.show(io::IO, node::CLAst.CLt)  = print(io, "<")
Base.show(io::IO, node::CLAst.CGt)  = print(io, ">")
Base.show(io::IO, node::CLAst.CLtE) = print(io, "<=")
Base.show(io::IO, node::CLAst.CGtE) = print(io, ">=")
Base.show(io::IO, node::CLAst.CEq)  = print(io, "==")
Base.show(io::IO, node::CLAst.CNotEq) = print(io, "!=")

Base.show(io::IO, node::CLAst.CAnd) = print(io, "&&")
Base.show(io::IO, node::CLAst.COr)  = print(io, "||")

Base.show(io::IO, node::CLAst.CNum)  = print(io, string(node.val))
Base.show(io::IO, node::CLAst.CName) = print(io, string(node.id))

clprint(io::IO, node::CLAst.CBoolOp, indent::Int) = begin
    print(io, "(")
    print(io, "$(node.values[1])")
    for val in node.values[2:end]
        print(io, " $(node.op) $val")
    end
    print(")")
end

clprint(io::IO, node::CLAst.CBinOp, indent::Int) = begin
    printind(io, "($(node.left) $(node.op) $(node.right))", indent)
end

clprint(io::IO, node::CLAst.CUnaryOp, indent::Int) = begin
    printind(io, "($(node.op)($(node.operand)))", indent)
end

clprint(io::IO, node::CLAst.CFunctionCall, indent::Int) = begin
    printind(io, "$(node.name)()", indent)
end

clprint(io::IO, node::CLAst.CBlock, indent::Int) = begin
    printind(io, "{{\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent + 1)
        print(io, ";\n")
    end
    printind(io, "}}\n", indent)
end

clprint(io::IO, node::CLAst.CAssign, indent::Int) = begin
    val = sprint() do io
        clprint(io, node.val, 0)
    end
    printind(io, "$(node.target) = $val", indent)
end

clprint(io::IO, node::CLAst.CAugAssignExpr, indent::Int) = begin
    print(io, "$(node.target) $(node.op)= $(node.value)")
end

clprint(io::IO, node::CLAst.CAssignExpr, indent::Int) = begin
    print(io, "$(node.targets[1]) = ")
    for target in node.targets[2:end]
        print(io, "$target")
    end
    print(io, "$(node.value)")
end

clprint(io::IO, node::CLAst.CStr, indent::Int) = begin
    print(io, "/"$(node.str)/"")
end

clprint(io::IO, node::CLAst.CNum, indent::Int) = begin
    print(io, "$(node.val)")
end

print_comma(io, i) = if i > 1; print(io, ", "); end

clprint(io::IO, node::CLAst.CTypeCast, indent::Int) = begin
    print(io, "($(node.ctype)) $(node.value)")
end

clprint(io::IO, node::CLAst.CLKernel, indent::Int) = begin
    print(io, "__kernel")
end

clprint(io::IO, node::CLAst.CIndex, indent::Int) = begin
    print(io, "$(node.val)")
end

clprint(io::IO, node::CLAst.CSubscript, indent::Int) = begin
    print(io, "$(node.val)[$(node.slice)")
end

clprint(io::IO, node::CLAst.CAttribute, indent::Int) = begin
    print(io, "$(node.val).$(node.attr)")
end

clprint(io::IO, node::CLAst.CPointerAttribute, indent::Int) = begin
    print(io, "$(node.val)-->$(node.attr)")
end

clprint(io::IO, node::CLAst.CIfExp, indent::Int) = begin
    print(io, "$(node.test) ? $(node.body) : $(node.orelse)")
end

printind(io::IO, str::String, indent::Int) = begin
    print(io, "\t"^indent, str)
end

clprint(io::IO, node::CLAst.CCompare, indent::Int) = begin
    print(io, "($(node.left)")
    for (op, right) in zip(node.ops, node.comparators)
        print(io, " $op $right")
    end
    print(io, ")")
end

clprint(io::IO, node::CLAst.CLModule, indent::Int) = begin
    printind(io, "// Automatically generated file! //\n\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent)
    end
end

clprint(io::IO, node::CLAst.CFunctionForwardDec, indent::Int) = begin
    printind(io, "$(node.return_type) $(node.name)($(node.args))", indent)
    printind("\n\n", indent)
end

clprint(io::IO, node::CLAst.CFunctionDef, indent::Int) = begin
    for decl in node.decl_list
        printind(io, "$decl\n", indent)
    end
    printind(io, "$(node.return_type) $(node.name)($(node.args))", indent)
    printind(io, "{{\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent + 1)
    end
    printind(io, "\n", indent)
    printind(io, "}}\n", indent)
end

clprint(io::IO, node::CLAst.CReturn, indent::Int) = begin
    if node.val == nothing
        printind(io, "return;\n", indent)
    else
        printind(io, "return($(node.val));\n", indent)
    end
end

clprint(io::IO, node::CLAst.CVarDec, indent::Int) = begin
    printind(io, "$(node.ctype) $(node.id);\n", indent)
end

clprint(io::IO, node::CLAst.CStruct, indent::Int) = begin
    printind(io, "typedef struct {{", indent)
    for decl in node.decl_list
        clprint(io, decl, indent + 1)
    end
    printind(io, "}} $(node.id);\n\n", indent)
end

clprint(io::IO, node::CLAst.CFor, indent::Int) = begin
    init = sprint() do io
        clprint(io, node.init, 0)
    end
    condition = sprint() do io
        clprint(io, node.condition, 0)
    end
    increment = sprint() do io
        clprint(io, node.increment, 0)
    end
    printind(io, "for ($init; $condition; $increment) {{\n", indent)
    for stmnt in node.block.body
        clprint(io, stmnt, indent + 1)
        print(io, ";\n") 
    end
    printind(io, "}}\n", indent)
end

clprint(io::IO, node::CLAst.CExpr, indent::Int) = begin
    printind(io, "$(node.val);\n", indent)
end

clprint(io::IO, node::CLAst.CIf, indent::Int) = begin
    printind(io, "if ($(node.test)) {{\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent + 1)
    end
    printind(io, "}}", indent)
    if node.orelse == nothing
        print(io, "\n")
    else
        for orelse in node.orelse
            printind(io, " else ", indent)
            if isa(orelse, CIf)
                clprint(io, orelse, indent)
            else
                printind(io, "{{", indent)
                clprint(io, orelse, indent + 1)
                printind(io, "}}\n", indent)
            end
        end
    end
end

clprint(io::IO, node::CLAst.CWhile, indent::Int) = begin
    printind(io, "while ($(node.test)) {{\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent + 1)
    end
    printind(io, "}}\n", indent)
end

clprint(io::IO, node::CLAst.CComment, indent::Int) = begin
    @assert !contains(node.str, "\n")
    printind(io, "// $(node.str) \n", indent)
end

clprint(io::IO, node::CLAst.CGroup, indent::Int) = begin
    for stmnt in node.body
        clprint(io, stmnt, indent)
    end
end

clprint(io::IO, node::CLAst.CBreak, indent::Int) = begin
    printind(io, "break;\n", indent)
end

clprint(io::IO, node::CLAst.CContinue, indent::Int) = begin
    printind(io, "continue;\n", indent)
end

end 
