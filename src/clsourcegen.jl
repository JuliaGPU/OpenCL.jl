module CLSourceGen


# see base/show.jl 292 for julia ast printing

Base.show(io::IO, node::CMult) = print(io,"*")
Base.show(io::IO, node::CAdd)  = print(io, "+")
Base.show(io::IO, node::CSub)  = print(io, "-")
Base.show(io::IO, node::CUSub) = print(io, "-")
Base.show(io::IO, node::CDiv)  = print(io, "/")
Base.show(io::IO, node::CMod)  = print(io, "%")
Base.show(io::IO, node::CNot)  = print(io, "!")
#TODO: bitwise

Base.show(io::IO, node::CLt)  = print(io, "<")
Base.show(io::IO, node::CGt)  = print(io, ">")
Base.show(io::IO, node::CLtE) = print(io, "<=")
Base.show(io::IO, node::CGtE) = print(io, ">=")
Base.show(io::IO, node::CEq)  = print(io, "==")
Base.show(io::IO, node::CNotEq) = print(io, "!=")

Base.show(io::IO, node::CAnd) = print(io, "&&")
Base.show(io::IO, node::COr)  = print(io, "||")

clprint(io::IO, node::CBoolOp, indent::Int) = begin
    print(io, "(")
    print(io, "$(node.values[1])")
    for val in node.values[2:end]
        print(io, " $(node.op) $val")
    end
    print(")")
end

clprint(io::IO, node::CAugAssignExpr, indent::Int) = begin
    print(io, "$(node.target) $(node.op)= $(node.value)")
end

clprint(io::IO, node::CAssignExpr, indent::Int) = begin
    print(io, "$(node.targets[1]) = ")
    for target in node.targets[2:end]
        print(io, "$target")
    end
    print(io, "$(node.value)")
end

clprint(io::IO, node::CStr, indent::Int) = begin
    print(io, "/"$(node.str)/"")
end

clprint(io::IO, node::CNum, indent::Int) = begin
    print(io, "$(node.n)")
end

print_comma(io, i) = if i > 1; print(io, ", "); end

clprint(io::IO, node::CTypeCast, indent::Int) = begin
    print(io, "($(node.ctype)) $(node.value)")
end

clprint(io::IO, node::CLKernel, indent::Int) = begin
    print(io, "__kernel")
end

clprint(io::IO, node::CIndex, indent::Int) = begin
    print(io, "$(node.val)")
end

clprint(io::IO, node::CSubscript, indent::Int) = begin
    print(io, "$(node.val)[$(node.slice)")
end

clprint(io::IO, node::CAttribute, indent::Int) = begin
    print(io, "$(node.val).$(node.attr)")
end

clprint(io::IO, node::CPointerAttribute, indent::Int) = begin
    print(io, "$(node.val)-->$(node.attr)")
end

clprint(io::IO, node::CIfExp, indent::Int) = begin
    print(io, "$(node.test) ? $(node.body) : $(node.orelse)")
end

printind(io::IO, str::String, indent::Int) = begin
    print(io, "\t"^indent, str)
end

clprint(io::IO, node::CCompare, indent::Int) = begin
    print(io, "($(node.left)")
    for (op, right) in zip(node.ops, node.comparators)
        print(io, " $op $right")
    end
    print(io, ")")
end

clprint(io::IO, node::CLModule, indent::Int) = begin
    printind(io, "// Automatically generated file! //\n\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent)
    end
end

clprint(io::IO, node::CFunctionForwardDec, indent::Int) = begin
    printind(io, "$(node.return_type) $(node.name)($(node.args))", indent)
    printind("\n\n", indent)
end

clprint(io::IO, node::CFunctionDef, indent::Int) = begin
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

clprint(io::IO, node::CReturn, indent::Int) = begin
    if node.val == nothing
        printind(io, "return;\n", indent)
    else
        printind(io, "return($(node.val));\n", indent)
    end
end

clprint(io::IO, node::CVarDec, indent::Int) = begin
    printind(io, "$(node.ctype) $(node.id);\n", indent)
end

clprint(io::IO, node::CStruct, indent::Int) = begin
    printind(io, "typedef struct {{", indent)
    for decl in node.decl_list
        clprint(io, decl, indent + 1)
    end
    printind(io, "}} $(node.id);\n\n", indent)
end

clprint(io::IO, node::CFor, indent::Int) = begin
    printind(io, "for ($(node.init); $(node.condition); $(node.increment)) {{\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent + 1)
    end
    printind(io, "\n", indent + 1)
    printind(io, "}}\n", indent)
end

clprint(io::IO, node::CExpr, indent::Int) = begin
    printind(io, "$(node.val);\n", indent)
end

clprint(io::IO, node::CIf, indent::Int) = begin
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

clprint(io::IO, node::CWhile, indent::Int) = begin
    printind(io, "while ($(node.test)) {{\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent + 1)
    end
    printind(io, "}}\n", indent)
end

clprint(io::IO, node::CComment, indent::Int) = begin
    @assert !contains(node.str, "\n")
    printind(io, "// $(node.str) \n", indent)
end

clprint(io::IO, node::CGroup, indent::Int) = begin
    for stmnt in node.body
        clprint(io, stmnt, indent)
    end
end

clprint(io::IO, node::CBreak, indent::Int) = begin
    printind(io, "break;\n", indent)
end

clprint(io::IO, node::CContinue, indent::Int) = begin
    printind(io, "continue;\n", indent)
end

end 
