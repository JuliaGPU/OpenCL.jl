module CLCompiler

using ..CLAst

typealias CLType Any
typealias CLInteger Union(Int16, Int32, Int64)

#TODO: run pass to convert goto statements into if / else statements

#TODO: pass to remove unnecessary array index casts

type GotoIfNot
    test 
    label
end

function rm_linenum!(expr::Expr)
    new_args = {}
    for ex in expr.args
        if is_linenumber(ex)
            continue
        end
        push!(new_args, ex)
    end
    for ex in new_args
        if isa(ex, Expr)
            rm_linenum!(ex)
        end
    end
    expr.args = new_args
    return expr
end

isconditional(n::GotoIfNot) = true
isconditional(n) = false

rm_goto_pass(ast::CAst) = begin
    for ex in ast.body
        if isconditional(ex)
            ifnode = CIf(ex.test, {}, {}, nothing)
            while !(isa(ex, LabelNode)) && ex.label != ex.label
                push!(ifnode.body, ex)
            end
            # skip node
        end
    end
end

visit(expr::Expr) = begin
    expr = rm_linenum!(expr)
    if haskey(visitors, expr.head)
        visitors[expr.head](expr)
    else
        error("unhandled head :$(expr.head)")
    end
end

visit(n::SymbolNode) = begin
    return CName(cname(n.name), n.typ)
end

visit(n::String) = begin
    error("unimplemented (visit) string")
end

visit(n::Number) = begin
    return CNum(n, typeof(n))
end

visit(n::GotoNode) = begin
    labelname = "label" * string(n.label)
    return CGoto(labelname)
end

visit(n::LabelNode) = begin
    labelname = "label" * string(n.label)
    return CLabel(labelname)
end

is_linenumber(ex::LineNumberNode) = true
is_linenumber(ex::Expr) = ex.head === :line
is_linenumber(ex) = false

ipointee_type{T}(::Type{Ptr{T}}) = T
array_type{T, N}(::Type{Array{T, N}}) = T

cname(s) = begin
    s = string(s)
    return s[1] == '#' ? s[2:end] : s
end

visit_block(expr::Expr) = begin
    @assert expr.head == :body || expr.head == :block
    body = {}
    for ex in expr.args
        if is_linenumber(ex)
            continue
        end
        push!(body, visit(ex))
    end
    return CBlock(body)
end

visit_gotoifnot(expr::Expr) = begin
    @assert expr.head == :gotoifnot
    node = visit(expr.args[1])
    return CIf(CUnaryOp(CNot(), node, Bool), 
               CGoto("label$(expr.args[2])"), 
               nothing)
    #return GotoIfNot(node, expr.args[2])
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
    target   = cname(expr.args[1])
    node     = visit(expr.args[2])
    ret_type = node.ctype 
    return CAssign(target, node, ret_type) 
end

visit_arrayset(expr::Expr) = begin
    @assert expr.args[1] === :arrayset
    target = visit(expr.args[2])
    val = visit(expr.args[3])
    idx = visit(expr.args[4])
    ty  = target.ctype
    return CAssign(CSubscript(target, CIndex(idx), ty),
                   val, ty)
end

visit_arrayref(expr::Expr) = begin
    @assert expr.args[1] == :arrayref
    #TODO: check that ref is a local var
    target = visit(expr.args[2])
    #TODO: do we need to pass around the global scope
    # as well?  we need to ensure 
    idx_node = visit(expr.args[3])
    ty = array_type(target.ctype)
    if isa(idx_node, CTypeCast)
        cast_ty = idx_node.ctype
        val_ty = idx_node.value.ctype
        if cast_ty != Uint32
            if val_ty == Uint32
                idx_node = idx_node.value
            else
                idx_node = CTypeCast(idx_node.value, Uint32)
            end
        end
    end
    return CSubscript(target, CIndex(idx_node), ty)
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
                                    :ult_int => CLt(),
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

const runtime_funcs = Set{Symbol}(:get_global_id)

visit_call(expr::Expr) = begin
    @assert expr.head === :call
    arg1 = first(expr.args)
    if arg1 === :arrayset
        return visit_arrayset(expr)
    end
    if arg1 === :arrayref
        return visit_arrayref(expr)
    end

    #TODO: handle runtime functions
    if arg1 in runtime_funcs
        args = CAst[]
        for arg in expr.args[2:end]
            push!(args, visit(arg))
        end
        return CFunctionCall(cname(arg1), args, Csize_t)
    end

    if !(isa(expr.args[1], TopNode))
        @show expr
        error("top node error")
    end
    
    # low level ccall functions
    if arg1.name === :ccall
        return visit_ccall(expr)
     
    # unbox boxed numbers
    elseif arg1.name === :box
        # expr.args[2] is a type symbol name
        ret_type = eval(expr.args[2])
        @assert isa(ret_type, DataType)
        if !(ret_type <: CLType)
            error("invalid cast to type $ret_type")
        end
        node = visit(expr.args[3])
        if !(promote_type(node.ctype, ret_type) == ret_type)
            @show node, ret_type
            error("ERR promote type")
        end

        if node.ctype === ret_type
            return node
        elseif isa(node, CNum)
            return CNum(node.val, ret_type)
        else
            return CTypeCast(node, ret_type)
        end
    
    # type assertions get translated to casts
    elseif arg1.name === :typeassert
        val = visit(expr.args[2])
        ty  = eval(expr.args[3])
        if val.ctype != nothing
            if isa(val.ctype, ty)
                # no op
                return val
            end
        end
        return CTypeCast(val, ty)

    # pow for integer exponents >= 4  
    elseif arg1.name === :power_by_squaring
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
    
    # less than if
    elseif arg1.name === :ltfsi64 || 
           arg1.name === :slt_int
        val = visit(expr.args[2])
        var = visit(expr.args[3])
        return CBinOp(val, CLt(), var, Cint)

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

visit_lambda(expr::Expr) = begin
    @assert expr.head === :lambda

    # parse variable declarations
    fargs = copy(expr.args[1])
    ctx = copy(expr.args[2])
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
            push!(args, CTypeDecl(cname(arg), ty))
        elseif ty <: Ptr
            push!(args, CPtrDecl(cname(arg), ty))
        elseif ty <: Array
            T = array_type(ty)
            push!(args, CPtrDecl(cname(arg), Ptr{T}))
        elseif ty === Any
            #TODO: look for unions in return types
            error("cannot compile type unstable function")
        else
            error("unhandled code path in visit_lambda parse_args")
        end
    end
    #reverse!(args)

    # predeclare local variables
    vars = CAst[]
    for var in localvars 
        ty = vartypes[var]
        if ty <: Number
            push!(vars, CVarDecl(cname(var), ty))
        elseif ty <: Ptr
            push!(vars, CPtrDecl(cname(var), ty))
        elseif ty <: Array
            #TODO: arrays need to be fixed size
            push!(vars, CArrayDecl(cname(var), ty))
        else
            @show var
            error("unknown code path in visit_lambda loc vars")
        end
    end

    # parse body
    blocknode = visit(expr.args[end])

    # prepend variable declarations in body
    prepend!(blocknode.body, reverse!(vars))
    
    # return type
    local ret_type::Type
    if isa(blocknode.body[end], CReturn)
        ret_type = blocknode.body[end].ctype
        if ret_type <: Array && ret_type != None
            T = array_type(ret_type)
            ret_type = Ptr{T}
        end
    else
        ret_type = Ptr{Void}
    end

    #TODO: function name?
    return CFunctionDef("testcl", args, blocknode, ret_type) 
end

const visitors = (Symbol=>Function)[:lambda => visit_lambda,
                                    :block  => visit_block,
                                    :body   => visit_block,
                                    :return => visit_return,
                                    :(=)    => visit_assign,
                                    :ref    => visit_index,
                                    :call   => visit_call,
                                    :gotoifnot => visit_gotoifnot,]
end
