module CLAst

abstract CAst 
abstract CType

type CMult  <: CAst end
type CAdd   <: CAst end
type CUAdd  <: CAst end
type CSub   <: CAst end
type CUSub  <: CAst end
type CDiv   <: CAst end
type CMod   <: CAst end
type CNot   <: CAst end
type CLt    <: CAst end 
type CGt    <: CAst end
type CLtE   <: CAst end
type CGtE   <: CAst end
type CEq    <: CAst end
type CNotEq <: CAst end
type CAnd   <: CAst end
type COr    <: CAst end 

#TODO: 
type CIndex <: CAst
end

#TODO: 
type CLModule <: CAst
end

#TODO: 
type CExpr <: CAst
end

type CFunctionCall <: CAst
    name
end

type CBlock <: CAst
    body
end

type CTypeCast <: CAst
    value
    ctype
end

type CVectorTypeCast <: CAst
    values
    ctype
end

type CWhile <: CAst
    test
    body
end

type CComment <: CAst
    str::String
end

type CGroup <: CAst
    body::Array{Any,1}
end

type CTypeName <: CAst
    typename
end

type CVarDec <: CAst
    id
    ctype
end

type CNum{T} <: CAst
    val::T
end

type CStr <: CAst
    val::String
    ctype
end

type CCall <: CAst
    func
    args
    keywords
    ctype
end

type CReturn <: CAst
    val
    ctype
end

type CFunctionForwardDec <: CAst
    name
    args
    return_type
end

type CFunctionDef <: CAst
    name
    args
    body
    return_type
end

type CName <: CAst
    id
    #ctx
    #ctype
end

type CBinOp <: CAst
    left
    op
    right
    #ctype
end

type CUnaryOp <: CAst
    op
    operand
    #ctype
end

type CKeyword <: CAst
    op
    operand
    ctype
end

type CLKernel <: CAst
end

type CSubscript
    val
    slice
    ctx
    ctype
end

abstract CAstAttribute 

type CAttribute <: CAstAttribute
    val
    attr
    ctx
    ctype
end

type CPointerAttribute <: CAstAttribute
    val
    attr
    ctx
    ctype
end

type CIf
    test
    body
    orelse
end

type CIfExp
    test
    body
    orelse
    ctype
end

type CCompare
    left
    ops
    comparators
    ctype
end

type CFor <: CAst
    init
    condition
    increment
    block
    orelse
end

CFor(init, cond, inc, body) = CFor(init, cond, inc, body, nothing)

type CStruct <: CAst
    id
    decl_list
end

type CAssign <: CAst
    target
    val
end

type CAssignExpr
    targets
    val
    ctype
end

type CAugAssignExpr 
    target
     op
    val
    #ctype
end

type CArray
    elts
    ctx
    ctype
end

type CBoolOp
    op
    vals
    ctype
end

type CContinue
end

type CBreak
end

end
