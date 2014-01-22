module CLCompiler

using ..CLAst

const builtins = (Symbol=>Function)[:add_int => identity]
const typemap = (Symbol => Type)[:Int64 => Int64]

typealias CLType Any
typealias CLInteger Union(Int16, Int32, Int64)

visit(expr::Expr) = begin
    if haskey(visitors, expr.head)
        visitors[expr.head](expr)
    else
        error("unhandled head :$(expr.head)")
    end
end

visit(n::SymbolNode) = begin
    #TODO: symbol with no type
    return CName(string(n.name), n.typ)
end

visit(n::Number) = begin
    return CNum(n, typeof(n))
end

visit_assign(expr::Expr) = begin
    @assert expr.head == :(=)
    target  = expr.args[1]
    val     = expr.args[2]
    return CAssign(target, val) 
end

#TODO: this only integer indices 
visit_index(expr::Expr) = begin
    @assert expr.head == :ref
    @assert typeof(expr.args[2]) <: Integer 
    return CIndex(expr.args[2])
end

visit_call(expr::Expr) = begin
    @assert expr.head == :call
    @assert isa(expr.args[1], TopNode)
    arg1 = first(expr.args)
    if arg1.name == :box
        ret_type = typemap[expr.args[2]]
        if !(ret_type <: CLType)
            error("invalid cast to type $ret_type")
        end
        node = visit(expr.args[3])
        @assert promote_type(node.ctype, ret_type) == ret_type 
        if node.ctype === ret_type
            return node
        else
            return CTypeCast(node, ret_type)
        end
    elseif arg1.name == :add_int
        lnode = visit(expr.args[2])
        rnode = visit(expr.args[3])
        ret_type = promote_type(lnode.ctype, rnode.ctype)
        if !(ret_type <: CLInteger)
            error("invalid return type for :add_int")
        end
        return CBinOp(lnode, CAdd(), rnode, ret_type)
    else
        error("invalid code path")
    end
end

const visitors = (Symbol=>Function)[:(=)  => visit_assign,
                                    :ref  => visit_index,
                                    :call => visit_call]
end
