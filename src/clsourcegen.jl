module CLSourceGen

using ..CLAst 

export clsource

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

printind(io::IO, str::String, indent::Int) = begin
    print(io, "\t"^indent, str)
end

pointee_type{T}(::Type{Ptr{T}}) = T

clprint{T}(io::IO, ::Ptr{T}, indent=Int) = begin
    ty = sprint() do io
        clprint(io, T, 0)
    end
    printind(io, "($ty *) ", indent)
end

clprint{T}(io::IO, node::Type{Ptr{T}}, indent=Int) = begin
    ty = sprint() do io
        clprint(io, T, 0)
    end
    printind(io, "($ty *)", indent)
end

clprint(io::IO, node::String, indent=Int) = begin
    printind(io, node, indent)
end

for (ty, cty) in [(:None, "void"),
                  (:Float64, "double"),
                  (:Float32, "float"),
                  (:Int64, "long"),
                  (:Uint64, "unsigned long")]
    @eval begin
        clprint(io::IO, node::Type{$ty}, indent::Int64) = begin
            printind(io, $("$cty"), indent)
        end
    end
end

clprint(io::IO, node::Float32, indent::Int) = begin
    @show node
    return node
end

clprint(io::IO, node::CLAst.CNum,  indent::Int) = begin
    val = sprint() do io
        show(io, node.val)
    end
    printind(io, "$val", indent)
end

clprint(io::IO, node::CLAst.CName, indent::Int) = begin
    printind(io, string(node.id), indent)
end

clprint(io::IO, node::CLAst.CBoolOp, indent::Int) = begin
    print(io, "(")
    print(io, "$(node.values[1])")
    for val in node.values[2:end]
        print(io, " $(node.op) $val")
    end
    print(")")
end

clprint(io::IO, node::CLAst.CBinOp, indent::Int) = begin
    left = sprint() do io
        clprint(io, node.left, 0)
    end
    right = sprint() do io
        clprint(io, node.right, 0)
    end
    printind(io, "($left $(node.op) $right)", indent)
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
    target = sprint() do io
        clprint(io, node.target, 0)
    end
    val = sprint() do io
        clprint(io, node.val, 0)
    end
    printind(io, "$target = $val", indent)
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
    val = sprint() do io
        show(io, node.val)
    end
    print(io, "$val")
end

print_comma(io, i) = if i > 1; print(io, ", "); end

clprint(io::IO, node::CLAst.CTypeCast, indent::Int) = begin
    ty = sprint() do io
        clprint(io, node.ctype, indent)
    end
    val = sprint() do io
        clprint(io, node.value, indent)
    end
    printind(io, "(($ty) $val)", indent)
end

clprint(io::IO, node::CLAst.CLKernel, indent::Int) = begin
    print(io, "__kernel")
end

clprint(io::IO, node::CLAst.CIndex, indent::Int) = begin
    val = sprint() do io
        clprint(io, node.val, 0)
    end
    printind(io, "$val", indent)
end

clprint(io::IO, node::CLAst.CSubscript, indent::Int) = begin
    val = sprint() do io
        clprint(io, node.val, 0)
    end
    idx = sprint() do io
        clprint(io, node.slice, 0)
    end
    printind(io, "$val[$idx]", indent)
end

clprint(io::IO, node::CLRTCall, indent::Int)= begin
    if node.args == nothing || length(node.args) == 0
        printind(io, "$(node.name)()", indent)
    else
        printind(io, "$(node.name)(", indent)
        nargs = length(node.args)
        for (i, arg) in enumerate(node.args)
            astr = sprint() do io
                clprint(io, arg, 0)
            end
            if i < nargs
                print(io, "$astr, ")
            else
                print(io, "$astr")
            end
        end
        print(") ")
    end
end

clprint(io::IO, node::CLAst.CPointerAttribute, indent::Int) = begin
    printind(io, "$(node.val)-->$(node.attr)", indent)
end

clprint(io::IO, node::CLAst.CIfExp, indent::Int) = begin
    printind(io, "$(node.test) ? $(node.body) : $(node.orelse)", indent)
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
    #TODO: decl list only for kernels?
    #for decl in node.decl_list
    #    printind(io, "$decl\n", indent)
    #end
    ret_type = sprint() do io
        clprint(io, node.ctype, 0)
    end
    printind(io, "$ret_type $(node.name)(", indent)
    nargs = length(node.args)
    for (i, arg) in enumerate(node.args)
        a = sprint() do io
            clprint(io, arg, 0)
        end
        if i < nargs
            print(io, "$a, ")
        else
            print(io, "$a)")
        end
    end
    clprint(io, node.body, indent)
end

clprint(io::IO, node::CLAst.CReturn, indent::Int) = begin
    if node.val == nothing
        printind(io, "return", indent)
    else
        val = sprint() do io
            clprint(io, node.val, 0)
        end
        printind(io, "return($val)", indent)
    end
end

clprint(io::IO, node::CLAst.CTypeDecl, indent::Int) = begin
    ty = sprint() do io
        clprint(io, node.ctype, 0)
    end
    printind(io, "(($ty) $(node.name))", indent)
end

clprint(io::IO, node::CLAst.CVarDecl, indent::Int) = begin
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

function clsource(n::CAst)
    return sprint() do io
        clprint(io, n, 0)
    end
end 

end

