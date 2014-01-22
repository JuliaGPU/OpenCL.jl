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
    return CName(string(n.name), n.typ)
end

visit(n::String) = begin
    error("unimplemented string")
end

visit(n::Number) = begin
    return CNum(n, typeof(n))
end

is_linenumber(ex::LineNumberNode) = true
is_linenumber(ex::Expr) = ex.head === :line
is_linenumber(ex) = false

visit_block(expr::Expr) = begin
    @assert expr.head == :body || expr.head == :block
    body = CAst[]
    for ex in expr.args
        if is_linenumber(ex)
            continue
        end
        push!(body, visit(ex))
    end
    return CBlock(body)
end

visit_return(expr::Expr) = begin
    @assert expr.head == :return
    @assert length(expr.args) == 1
    if expr.args[1] != nothing
        node = visit(expr.args[1])
        return CReturn(node, node.ctype)
    else
        return CReturn(nothing, Void)
    end
end

visit_assign(expr::Expr) = begin
    @assert expr.head == :(=)
    target   = string(expr.args[1])
    node     = visit(expr.args[2])
    ret_type = node.ctype 
    return CAssign(target, node, ret_type) 
end

#TODO: this only integer indices 
visit_index(expr::Expr) = begin
    @assert expr.head == :ref
    @assert typeof(expr.args[2]) <: Integer 
    return CIndex(expr.args[2])
end

const builtin_funcs = (Symbol => String) [:pow => "pow",
                                          :powf => "pow"] 

function call_builtin(fname, expr::Expr, ret_type::Type)
    if !haskey(builtin_funcs, fname)
        error("unknown builtin function $fname")
    end
    if fname === :pow || fname === :powf
        arg1 = visit(expr.args[5])
        arg2 = visit(expr.args[7])
        ret_type = promote_type(arg1.ctype, arg2.ctype)
        return CLRTCall("pow", [arg1, arg2], ret_type)  
    end
end

visit_ccall(expr::Expr) = begin
    @assert isa(expr.args[1], TopNode)
    @assert expr.args[1].name == :ccall
    # get function name
    fname    = (expr.args[2].args[2])
    @assert isa(fname, QuoteNode)
    # get function return type
    ret_type = eval(expr.args[3])
    @assert isa(ret_type, DataType) 
    return call_builtin(fname.value, expr, ret_type)
end

# builtins
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
    @assert expr.head === :call
    @assert isa(expr.args[1], TopNode)
    arg1 = first(expr.args)
    
    # low level ccall functions
    if arg1.name === :ccall
        return visit_ccall(expr)

    # pow for integer exponents >= 4  
    elseif arg1.name == :power_by_squaring
        arg1 = visit(expr.args[2])
        arg2 = visit(expr.args[3])
        ret_type = promote_type(arg1.ctype, arg2.ctype)
        if arg2.ctype <: Integer
            return CLRTCall("pown", [arg1, arg2], ret_type) 
        elseif arg2.ctype <: FloatingPoint
            return CLFTCall("pow", [arg1, arg2], ret_type)
        else
            error("invalid code path in power_by_squaring")
        end

    # unbox boxed numbers
    elseif arg1.name == :box
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
        elseif isa(node, CNum)
            return CNum(node.val, ret_type)
        else
            return CTypeCast(node, ret_type)
        end

    # cast integer to float 
    elseif arg1.name === :sitofp
        ty = expr.args[2]
        @assert ty <: Number
        n = visit(expr.args[3])
        if isa(n, Number)
            return CNum(n, ty)
        elseif isa(n, CNum)
            return CNum(convert(ty, n.val), ty)
        elseif isa(n, CName)
            n.ctype = ty
            return n
        else
            error("invalid code path in :sitofp")
        end

    # cast signed /unsigned integers
    elseif (arg1.name === :sext_int ||
            arg1.name === :zext_int)
        node = visit(expr.args[3]) 
        return node

    # cast fp 
    elseif arg1.name === :fpext
        node = visit(expr.args[3])
        return node
    
    # truncated fp casting
    elseif arg1.name === :fptrunc
        ret_type =  eval(expr.args[2])
        @assert isa(ret_type, DataType)
        node = visit(expr.args[3])
        return CTypeCast(node, ret_type)
    
    # binary operations
    elseif haskey(binary_builtins, arg1.name) 
        op = binary_builtins[arg1.name]
        lnode = visit(expr.args[2])
        rnode = visit(expr.args[3])
        ret_type = promote_type(lnode.ctype, rnode.ctype)
        return CBinOp(lnode, op, rnode, ret_type)
    
    # unary operations
    elseif haskey(unary_builtins, arg1.name)
        op = unary_builtins[arg1.name]
        node = visit(expr.args[3])
        ret_type = node.ctype
        return CUnaryOp(op, node, ret_type)
    else
        @show expr
        error("unhandled call function :$(arg1)")
    end
end

pointee_type{T}(::Type{Ptr{T}}) = T

visit_lambda(expr::Expr) = begin
    @assert expr.head === :lambda

    # parse variable declarations
    fargs = Set{Symbol}(expr.args[1]...)
    ctx = expr.args[2]
    localvars = Set{Symbol}(ctx[1]...)
    vartypes  = (Symbol => Type)[]
    for var in ctx[2]
        vartypes[var[1]] = var[2]
    end
    
    # parse args
    args = CAst[]
    for arg in fargs
        ty = vartypes[arg]
        if ty <: Number 
            push!(args, CTypeDecl(string(arg), ty))
        else ty <: Ptr
            push!(args, CPtrDecl(string(arg), ty))
        end
    end

    # parse body
    blocknode = visit(expr.args[end])
    
    # return type
    local ret_type::Type
    if isa(blocknode.body[end], CReturn)
        ret_type = blocknode.body[end].ctype
    else
        ret_type = Ptr{Void}
    end
 
    @show args
    @show typeof(localvars)
    @show vartypes
    @show blocknode
    @show ret_type
    return CFunctionDef("testx", args, blocknode, ret_type) 
end

const visitors = (Symbol=>Function)[:lambda => visit_lambda,
                                    :block  => visit_block,
                                    :body   => visit_block,
                                    :return => visit_return,
                                    :(=)    => visit_assign,
                                    :ref    => visit_index,
                                    :call   => visit_call]
end
