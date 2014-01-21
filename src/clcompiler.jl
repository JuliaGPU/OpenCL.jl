module CLCompiler

const visitors = (Symbol=>Function){:(=) => visit_assign,
                                    :ref => visit_index}

visit_assign(expr::Expr) = begin
    @assert expr.head == :(=)
    target  = expr.args[1]
    val     = expr.args[2]
    return CAssign(target, val) 
end

#TODO: this only integer indices 
visit_index(expr::Expr) = begin
    @assert expr.head = :ref
    @assert typeof(expr.args[2]) <: Integer 
    return CIndex(expr.args[2])
end


# function test1(x::Int64)
#              return x + 1
#          end

# int testx(int64_t x) {
#     return x + 1;
# }
