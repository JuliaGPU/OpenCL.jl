module CLCompiler2

using ..CLAst

export visit

typealias CLType Any
typealias CLInteger Union(Int16, Int32, Int64)
typealias CLInt Int32

clint(x) = convert(Cint, x)
clfloat(x) = convert(Cfloat, x)

ctype{T}(x::Range{T}) = T

visit(node::Expr) = begin
    if haskey(visitors, node.head)
        visitors[node.head](node)
    else
        error("unhandled head :$(node.head)")
    end
end

visit(node::Symbol) = begin
    return CName(string(node), nothing)
end

visit_for(expr::Expr) = begin
    @assert expr.head == :for
    node = visit(expr.args[1])
    if isa(node, CAssign) && isa(node.val, Range)
        name = node.target.name
        ty   = node.target.ctype
        ast = CFor(CAssign(node.target,
                           CNum(node.val.start), 
                           ty),
                   CBinOp(name,
                          CLtE(), 
                          CNum(node.val.len),
                          Bool),
                   CAssign(name,
                           CBinOp(name,
                                  CAdd(),
                                  CNum(node.val.step),
                                  ty),
                           ty),
                   CBlock([]))
     end
end

visit_block(expr::Expr) = begin
    @assert expr.head == :block
    @show expr.args
end

visit_assign(expr::Expr) = begin
    @assert expr.head == :(=)
    @assert isa(expr.args[1], Symbol)
    name = string(expr.args[1])
    val  = visit(expr.args[2])
    ty   = ctype(val)
    target = CVarDecl(CName(name, ty), ty)
    return CAssign(target, val, ty) 
end

are_integers(xs...) = begin
    for x in xs
        if !(isa(x, Integer))
            return false
        end
    end
    return true
end

visit_colon(expr::Expr) = begin
    @assert expr.head == :(:)
    if length(expr.args) == 2
        if !(are_integers(expr.args...))
            error("only int range args are supported")
        end
        return Range{CLInt}(clint(expr.args[1]),
                            clint(1),
                            clint(expr.args[2]))
    elseif length(expr.args) == 3
        if !(are_integers(expr.args...))
            error("only int range args are supported")
        end
        return Range{CLInt}(clint(expr.args[1]),
                            clint(expr.args[2]),
                            clint(expr.args[3]))
    else
        error("invalid colon expression")
    end
end

const visitors = (Symbol=>Function)[:for    => visit_for,
                                    :(=)    => visit_assign,
                                    :(:)    => visit_colon,
                                    :block  => visit_block]
end
