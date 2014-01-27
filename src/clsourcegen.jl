module CLSourceGen

using ..CLAst 

export clsource

# see base/show.jl 292 for julia ast printing

clprint(io::IO, node::CLAst.CMult, indent::Int) = print(io,"*")
clprint(io::IO, node::CLAst.CAdd, indent::Int)  = print(io, "+")
clprint(io::IO, node::CLAst.CUAdd, indent::Int) = print(io, "++")
clprint(io::IO, node::CLAst.CSub, indent::Int)  = print(io, "-")
clprint(io::IO, node::CLAst.CUSub, indent::Int) = print(io, "--")
clprint(io::IO, node::CLAst.CDiv, indent::Int)  = print(io, "/")
clprint(io::IO, node::CLAst.CMod, indent::Int)  = print(io, "%")
clprint(io::IO, node::CLAst.CNot, indent::Int)  = print(io, "!")
#TODO: bitwise

clprint(io::IO, node::CLAst.CLt, indent::Int)  = print(io, "<")
clprint(io::IO, node::CLAst.CGt, indent::Int)  = print(io, ">")
clprint(io::IO, node::CLAst.CLtE, indent::Int) = print(io, "<=")
clprint(io::IO, node::CLAst.CGtE, indent::Int) = print(io, ">=")
clprint(io::IO, node::CLAst.CEq, indent::Int)  = print(io, "==")
clprint(io::IO, node::CLAst.CNotEq, indent::Int) = print(io, "!=")

clprint(io::IO, node::CLAst.CAnd, indent::Int) = print(io, "&&")
clprint(io::IO, node::CLAst.COr, indent::Int)  = print(io, "||")

#TODO: bit shift operations

#Base.show(io::IO, node::CLAst.CNum)  = print(io, string(node.val))
#Base.show(io::IO, node::CLAst.CName) = print(io, string(node.id))

printind(io::IO, str::String, indent::Int) = begin
    print(io, "\t"^indent, str)
end

pointee_type{T}(::Type{Ptr{T}}) = T

clprint{T}(io::IO, ::Ptr{T}, indent=Int) = begin
    ty = sprint() do io
        clprint(io, T, 0)
    end
    printind(io, "$ty *", indent)
end

clprint{T}(io::IO, node::Type{Ptr{T}}, indent=Int) = begin
    ty = sprint() do io
        clprint(io, T, 0)
    end
    printind(io, "$ty *", indent)
end

clprint(io::IO, node::String, indent=Int) = begin
    printind(io, node, indent)
end

for (ty, cty) in [(:None, "void"),
                  (:Float64, "double"),
                  (:Float32, "float"),
                  (:Uint32, "unsigned int"),
                  (:Int64, "long"),
                  (:Uint64, "unsigned long")]
    @eval begin
        clprint(io::IO, node::Type{$ty}, indent::Int64) = begin
            printind(io, $("$cty"), indent)
        end
    end
end

clprint{T}(io::IO, node::Type{Range{T}}, indent::Int) = begin
    printind(io, "Range", indent)
end

clprint(io::IO, node::Type{(NTuple{2, Int64})}, indent::Int) = begin
    printind(io, "int2", indent)
end

clprint(io::IO, node::CLAst.CArray, indent::Int) = begin
    printind(io, "{", indent)
    nelts = length(node.elts)
    for (i, n) in enumerate(node.elts)
        el = sprint() do io
            clprint(io, n, 0)
        end
        if i < nelts
            printind(io, "$el,", 0)
        else
            printind(io, "$el}", 0)
        end
    end
end

clprint(io::IO, node::CLAst.CStructRef, indent::Int) = begin
    name = sprint() do io
        clprint(io, node.name, 0)
    end
    field = sprint() do io
        clprint(io, node.field, 0)
    end
    printind(io, "$name.$field", indent)
end

clprint(io::IO, node::Float64, indent::Int) = begin
    printind(io, string(node), 0)
end

clprint(io::IO, node::Float32, indent::Int) = begin
    printind(io, string(node) * "f", 0)
end

clprint(io::IO, node::Int64, indent::Int) = begin
    printind(io, string(node), 0)
end

clprint(io::IO, node::Uint64, indent::Int) = begin
    printind(io, string(node) * "u", 0)
end

clprint{T}(io::IO, node::CLAst.CNum{T},  indent::Int) = begin
    clprint(io, node.val, 0)
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
    op = sprint() do io
        clprint(io, node.op, 0)
    end
    right = sprint() do io
        clprint(io, node.right, 0)
    end
    printind(io, "$left $op $right", indent)
end

clprint(io::IO, node::CLAst.CUnaryOp, indent::Int) = begin
    op = sprint() do io
        clprint(io, node.op, 0)
    end
    operand = sprint() do io
        clprint(io, node.operand, 0)
    end
    printind(io, "$op($operand)", indent)
end

clprint(io::IO, node::CLAst.CFunctionCall, indent::Int) = begin
    printind(io, "$(node.name)(", indent)
    if node.args != nothing
        nargs = length(node.args)
        for (i, arg) in enumerate(node.args)
            a = sprint() do io
                clprint(io, arg, 0)
            end
            printind(io, a, 0)
            if i < nargs
                printind(io, ", ", 0)
            end
        end
    end
    printind(io, ")", 0)
end

clprint(io::IO, node::CLAst.CBlock, indent::Int) = begin
    printind(io, "{{\n", indent)
    for stmnt in node.body
        clprint(io, stmnt, indent + 1)
        print(io, ";\n")
    end
    printind(io, "}}\n", indent)
end

clprint(io::IO, node::CLAst.CAssignList, indent::Int) = begin
    nassign = length(node.list)
    for (i, a) in enumerate(node.list)
        clprint(io, a, indent);
        if i < nassign
            printind(io, ";\n", 0)
        end
    end
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
        print(io, ")")
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
    print(io, "typedef struct Range {long start; long step; long len; } Range;\n")

    printind(io, "__kernel $ret_type $(node.name)(\n\t", indent)
    nargs = length(node.args)
    for (i, arg) in enumerate(node.args)
        a = sprint() do io
            clprint(io, arg, 0)
        end
        if i < nargs
            print(io, "$a,\n\t")
        else
            print(io, "$a)\n")
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

clprint(io::IO, node::CLAst.CPtrDecl, indent::Int) = begin
    ty = sprint() do io
        clprint(io, node.ctype, 0)
    end
    printind(io, "__global $ty$(node.name)", indent)
end

clprint(io::IO, node::CLAst.CTypeDecl, indent::Int) = begin
    ty = sprint() do io
        clprint(io, node.ctype, 0)
    end
    printind(io, "$ty $(node.name)", indent)
end

#TODO: Array Decl
clprint(io::IO, node::CLAst.CVarDecl, indent::Int) = begin
    ty = sprint() do io 
        clprint(io, node.ctype, 0)
    end
    printind(io, "$ty $(node.name)", indent)
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
    test = sprint() do io
        clprint(io, node.test, 0)
    end
    ifbody = sprint() do io
        clprint(io, node.body, 0)
    end
    printind(io, "if ($test) ", indent)
    printind(io, ifbody, 0)
    if node.orelse != nothing
        printind(io, "else ", indent)
        elsebody = sprint() do io
            clprint(io, node.orelse, 0)
        end
        printind(io, elsebody, 0)
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
    printind(io, "break", indent)
end

clprint(io::IO, node::CLAst.CContinue, indent::Int) = begin
    printind(io, "continue", indent)
end

clprint(io::IO, node::CLAst.CLabel, indent::Int) = begin
    printind(io, node.name * ":", indent)
end

clprint(io::IO, node::CLAst.CGoto, indent::Int) = begin
    printind(io, "goto " * node.label, indent)
end

function clsource(n::CAst)
    return sprint() do io
        #println(io, "#pragma OPENCL EXTENSION cl_khr_fp64: enable")
        clprint(io, n, 0)
    end
end 

end

