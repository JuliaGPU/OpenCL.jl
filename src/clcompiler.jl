module CLCompiler

using ..CLAst


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

#ccall, cglobal, abs_float, add_float, add_int, and_int, ashr_int,
#box, bswap_int, checked_fptosi, checked_fptoui, checked_sadd,
#checked_smul, checked_ssub, checked_uadd, checked_umul, checked_usub,
#nan_dom_err, copysign_float, ctlz_int, ctpop_int, cttz_int,
#div_float, eq_float, eq_int, eqfsi64, eqfui64, flipsign_int, select_value,
#fpext64, fpiseq, fpislt, fpsiround, fpuiround, fptosi, fptoui,
#fptrunc32, le_float, lefsi64, lefui64, lesif64,
#leuif64, lshr_int, lt_float, ltfsi64, ltfui64, ltsif64, ltuif64, mul_float,
#mul_int, ne_float, ne_int, neg_float, neg_int, not_int, or_int, rem_float,
#sdiv_int, shl_int, sitofp, sle_int, slt_int, smod_int,
#srem_int, sub_float, sub_int, trunc_int, udiv_int, uitofp,
#ule_int, ult_int, unbox, urem_int, xor_int, sext_int, zext_int

const binary_builtins = (Symbol=>CAst)[
                                    :add_int => CAdd(),
                                    :add_float => CAdd(),
                                    :div_lfoat => CDiv(),
                                    :eq_float => CEq(),
                                    :eq_int => CEq(),
                                    :le_float => CLtE(),
                                    :le_int => CLtE(),
                                    :lt_float => CLt(),
                                    :lt_int => CLt(),
                                    :mul_float => CMult(),
                                    :mul_int => CMult(),
                                    :ne_float => CNotEq(),
                                    :ne_int => CNotEq(),
                                    :not_int => CNot(),
                                    :or_int => COr(), 
                                    :smod_int => CMod()]

const unary_builtins = (Symbol=>CAst)[
                                    :neg_float => CUSub(),
                                    :neg_int => CUSub()]
visit_call(expr::Expr) = begin
    @assert expr.head == :call
    @assert isa(expr.args[1], TopNode)
    arg1 = first(expr.args)
    if arg1.name == :box
        # expr.args[2] is a type symbol name
        ret_type = eval(expr.args[2])
        @assert isa(ret_type, DataType)
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
    # cast integer to float 
    elseif arg1.name == :sitofp
        ty = expr.args[2]
        @assert typeof(expr.args[3]) <: Integer
        return CNum(convert(ty, expr.args[3]))
    # cast signed /unsigned integers
    elseif (arg1.name == :sext_int ||
            arg1.name == :zext_int)
        node = visit(expr.args[3]) 
        return node
    elseif haskey(binary_builtins, arg1.name) 
        op = binary_builtins[arg1.name]
        lnode = visit(expr.args[2])
        rnode = visit(expr.args[3])
        ret_type = promote_type(lnode.ctype, rnode.ctype)
        #TODO: return type checking
        return CBinOp(lnode, CAdd(), rnode, ret_type)
    elseif haskey(unary_builtins, arg1.name)
        op = unary_builtins[arg1.name]
        node = visit(expr.args[3])
        ret_type = node.ctype
        return CUnaryOp(op, node, ret_type)
    else
        error("unhandled call function :$(arg1)")
    end
end

const visitors = (Symbol=>Function)[:(=)  => visit_assign,
                                    :ref  => visit_index,
                                    :call => visit_call]
end
