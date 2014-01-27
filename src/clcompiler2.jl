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

is_linenumber(ex::LineNumberNode) = true
is_linenumber(ex::Expr) = ex.head === :line
is_linenumber(ex) = false

const unops  = (Symbol => CAst)[:(!)  => CNot()]

const binops = (Symbol => CAst)[:(*)  => CMult(),
                                :(+)  => CAdd(),
                                :(-)  => CSub(),
                                :(/)  => CDiv(),
                                :(%)  => CMod(),
                                :(==) => CEq(),
                                :(!=) => CNotEq(),
                                :(<)  => CLt(),
                                :(<=) => CLtE(),
                                :(>)  => CGt(),
                                :(>=) => CGtE(),
                                :(>>) => CBitShiftRight(),
                                :(<<) => CBitShiftLeft()]


visit(node::Symbol) = begin
    return CName(string(node), nothing)
end

visit(node::Number) = begin
    if isa(node, Integer)
        if isa(node, Int)
            return CNum(clint(node))
        else
            return CNum(node)
        end
    elseif isa(node, FloatingPoint)
        return CNum(node)
    else
        error("Unhandled CNum type $(typeof(node))")
    end
end

visit_call(expr::Expr) = begin
    @assert expr.head == :call
    arg1 = expr.args[1]
    if haskey(unops, arg1)
        op = unops[arg1]
        node = visit(expr.args[2])
        #TODO: correct types
        if isa(op, CNot)
            return CUnaryOp(op, node, Bool)
        else
            error("unhandled code path")
        end
    end
    if haskey(binops, arg1)
        op = binops[arg1]
        arg1 = visit(expr.args[2])
        arg2 = visit(expr.args[3])
        return CBinOp(arg1, op, arg2, arg1.ctype)
    end
    error("unhandled code path in call")
end

visit_and(expr::Expr) = begin
    @assert expr.head == :(&&)
    arg1 = visit(expr.args[1])
    arg2 = visit(expr.args[2])
    return CBinOp(arg1, CAnd(), arg2, Bool)
end

visit_or(expr::Expr) = begin 
    @assert expr.head == :(||)
    arg1 = visit(expr.args[1])
    arg2 = visit(expr.args[2])
    return CBinOp(arg1, COr(), arg2, Bool)
end

visit_add(expr::Expr) = begin
    @assert expr.head == :(+=)
    arg1 = visit(expr.args[1])
    arg2 = visit(expr.args[2])
    return CAssign(arg1,
                   CBinOp(arg1, CAdd(), arg2, nothing),
                   nothing)
end

visit_sub(expr::Expr) = begin
    @assert expr.head == :(-=)
    arg1 = visit(expr.args[1])
    arg2 = visit(expr.args[2])
    return CAssign(arg1,
                   CBinOp(arg1, CSub(), arg2, nothing),
                   nothing)
end

visit_mult(expr::Expr) = begin
    @assert expr.head == :(*=)
    arg1 = visit(expr.args[1])
    arg2 = visit(expr.args[2])
    return CAssign(arg1,
                   CBinOp(arg1, CMult(), arg2, nothing),
                   nothing)

end

visit_div(expr::Expr) = begin
    @assert expr.head == :(/=)
    arg1 = visit(expr.args[1])
    arg2 = visit(expr.args[2])
    return CAssign(arg1,
                   CBinOp(arg1, CDiv(), arg2, nothing),
                   nothing)
end

visit_comparison(expr::Expr) = begin
    @assert expr.head == :comparison
    sym_cmp = expr.args[2]
    if !(haskey(binops, sym_cmp))
        error("Unknown comparison $sym_cmp")
    end
    arg1 = visit(expr.args[1])
    cmp  = binops[sym_cmp]
    arg2 = visit(expr.args[3])
    return CBinOp(arg1, cmp, arg2, Bool)
end

visit_if(expr::Expr) = begin
    @assert expr.head == :if
    cond   = visit(expr.args[1])
    block  = visit(expr.args[2])
    if length(expr.args) == 3
        orelse = visit(expr.args[3])
    else
        orelse = nothing
    end
    return CIf(cond, block, orelse) 
end

visit_while(expr::Expr) = begin
    @assert expr.head == :while
    node  = visit(expr.args[1])
    block = visit(expr.args[2])
    return CWhile(node, block)
end

visit_for(expr::Expr) = begin
    @assert expr.head == :for
    node = visit(expr.args[1])
    body = visit(expr.args[2])
    if isa(node, CAssign) && isa(node.val, Range)
        name = node.target.name
        ty   = node.target.ctype
        init = CAssign(node.target, CNum(node.val.start), ty)
        cond = CBinOp(name, CLtE(), CNum(node.val.len), Bool)
        incr = CAssign(name, CBinOp(name, CAdd(), CNum(node.val.step), ty), ty)
        block  = body
        return CFor(init, cond, incr, block)
    else
        error("unhandled for path")
    end
end

visit_block(expr::Expr) = begin
    @assert expr.head == :block
    if length(expr.args) == 0
        return CBlock([])
    else
        body = CAst[]
        for node in expr.args
            if !(is_linenumber(node))
                push!(body, visit(node))
            end
        end
        return CBlock(body)
    end
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

const visitors = (Symbol=>Function)[:call   => visit_call,
                                    :comparison => visit_comparison,
                                    :for    => visit_for,
                                    :if     => visit_if,
                                    :(=)    => visit_assign,
                                    :(:)    => visit_colon,
                                    :(&&)   => visit_and,
                                    :(||)   => visit_or,
                                    :+=     => visit_add,
                                    :-=     => visit_sub,
                                    :*=     => visit_mult,
                                    :/=     => visit_div,
                                    :block  => visit_block,
                                    :while  => visit_while]
end
