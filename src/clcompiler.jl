module CLCompiler

import ..CLAst CAssign, CIndex, CTypeCast

const visitors = (Symbol => Function){:(=)  => visit_assign,
                                      :ref  => visit_index,
                                      :call => visit_call}

const builtins = (Symbol => Function){:add_int => nothing}

visit(expr::Expr) = begin
    if haskey(visitors, expr.head)
        visitors[expr.head](expr)
    else
        error("unhandled head :$(expr.head)")
    end
end

visit(node::SymbolNode) = begin
    return CName(node.name, node.typ)
end

visit(node::Number) = begin
    return CNumber(expr, typeof(expr))
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
    @assert expr.args[1] <: TopNode
    arg1 = first(expr.args)
    if arg1.name == :box
        ret_type = expr.args[2]
        if !(ret_type) <: CLType
            error("invalid cast to type $ret_type")
        end
        node = visit(expr.args[3])
        @assert promote_type(node.ctype, ret_type) == ret_type 
        return CTypeCast(node, ret_type)
    elseif arg1.name == :add_int
        lnode = visit(arg1.args[2])
        rnode = visit(arg1.args[3])
        ret_type = promote_type(lnode.ctype, rnode.ctype)
        if !(isa(ret_type, CLInt))
            error("invalid return type for :add_int")
        end
        return CBinOp(lnode, CAdd(), rnode, ret_type)
    else
        error("invalid code path")
    end
end

# function test1(x::Int64)
#              return x + 1
#          end

# int testx(int64_t x) {
#     return x + 1;
# }
