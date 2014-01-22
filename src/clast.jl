module CLAst

export CAst, CAssign, CBlock, CIndex, CTypeCast, CName, CNum, CBinOp,
       CMult, CAdd, CLt, CLtE, CDiv, CEq, CNotEq, CNot, COr, CMod,
       CUSub, CUnaryOp, CUAdd, CFunctionCall, CFor, CReturn,
       CSubscript, CLRTCall

abstract CAst 
abstract CType

Base.isequal(a1::CAst, a2::CAst) = begin
    if typeof(a1) != typeof(a2)
        return false
    end
    for field in names(a1)
        if getfield(a1, field) != getfield(a2, field)
            return false
        end
    end
    return true
end

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
    val
    ctype
end

CIndex(val) = CIndex(val, Csize_t)

#TODO: 
type CLModule <: CAst
end

#TODO: 
type CExpr <: CAst
end

type CLRTCall <: CAst
    name
    args
    ctype
end

type CFunctionCall <: CAst
    name
    args
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
    ctype::Type
end

type CNum{T} <: CAst
    val::T
    ctype::Type{T}
end
CNum{T}(x::T) = CNum{T}(x, T)
CNum(x, T) = CNum{T}(convert(T, x), T)

type CStr <: CAst
    val::String
    ctype::Ptr{Cchar}
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
    ctype
end

CName(id) = CName(id, nothing)

type CBinOp <: CAst
    left
    op
    right
    ctype::Type
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
    #ctx
    #ctype
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
    ctype
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
