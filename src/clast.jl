abstract CAst 

type CTypeCast <: CAst
    value
    ctype
end

type CVectorTypeCast <: CAst
    values
    ctype
end

type CComment <: CAst
    str::String
end

type CGroup <: CAst
    body
end

type CTypeName <: CAst
    typename
end

type CVarDec <: CAst
    id
    ctype
end

type CNumber{T} <: CAst
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
    ctx
    ctype
end

type CBinOp <: CAst
    left
    op
    right
    ctype
end

type CUnaryOp <: CAst
    op
    operand
    ctype
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
    body
    orelse
end

type CStruct <: CAst
    id
    decl_list
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
    ctype
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

