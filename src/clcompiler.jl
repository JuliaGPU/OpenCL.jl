module Compiler

using DataStructures
using ..CLAst
using ..SourceGen

export build_kernel, visit, structgen!

typealias CLScalarTypes Union(Bool,
                              Int8,
                              Uint8,
                              Int16,
                              Uint16,
                              Int32,
                              Uint32,
                              Int64,
                              Uint64,
                              Int128,
                              Uint128,
                              Uint64,
                              Float16,
                              Float32,
                              Float64,
                              Complex64,
                              Complex128,
                              Void)
type CLModule
    pragmas::Set
    constants::Set
    structs::OrderedDict{Type, CStruct}
    functions::OrderedDict{Symbol, CFunctionDef}
    kernels::OrderedDict{Symbol, CLKernelDef}
end

CLModule() = CLModule(Set(),
                      Set(),
                      OrderedDict{Type, CStruct}(),
                      OrderedDict{Symbol, CFunctionDef}(),
                      OrderedDict{Symbol, CLKernelDef}())
type CLContext
    func_args::Array
    local_vars::Set{Symbol}
    var_types::Dict{Symbol, Type}
    funcs::OrderedDict{Symbol, CAst}
    structs::Dict{Type, CAst}
    extensions::Array
end

CLContext() = CLContext({}, 
                      Set{Symbol}(), 
                      Dict{Symbol, Type}(), 
                      OrderedDict{Symbol, CAst}(),
                      Dict{Type, CAst}(),
                      {})

typealias CLType Any
typealias CLInteger Union(Int16, Int32, Int64)

pointer_type{T}(::Type{Ptr{T}}) = T
array_elemtype{T,N}(::Type{Array{T, N}}) = T
range_elemtype{T}(::Type{Range1{T}}) = T
range_elemtype{T}(::Type{Range{T}}) = T

# TODO: this
cname(s) = begin
    s = string(s)
    return s[1] == '#' ? s[2:end] : s
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

# TODO: this is incomplete but it works for now
function cstruct_name{T}(::Type{T})
    s = split(string(T), ['{', ',', '}'], false)
    return join(s, "_")
end

function isvalid_clstruct(ty::DataType, parent_type::Type=None)
    if !(Base.isstructtype(T))
        error("structgen error, type $T is not a valid struct type")
    end
    if length(ty.types) == 0
        error("type $ty has no fields")
    end
    for (fname, fty) in zip(names(ty), ty.types)
        if ty === fty || ty === parent_type
            error("c struct fields cannot have self referential struct types")
        end
        if Base.isstructtype(fty)
           isvalid_clstruct(fty, ty)
        elseif fty <: Ptr
            if !(pointer_type(fty) <: CLScalarTypes)
                error("c struct field $fname has invalid CLType $fty")
            end
        elseif !(fty <: CLScalarTypes)
                error("c struct field $fname has invalid CLType $fty")
        end
   end
   return true
end

function structgen!{T}(clmod::CLModule, ::Type{T})
    decl_list = CAst[]
    for (name, ty) in zip(names(T), T.types)
        if Base.isstructtype(ty)
            if !haskey(clmod.structs, ty)
                structgen!(clmod, ty)
            end
        end
        push!(decl_list, CTypeDecl(cname(name), ty))
    end
    sname = cstruct_name(T) 
    @assert haskey(clmod.structs, T) == false
    clmod.structs[T] = CStruct(sname, decl_list)
    return 
end

visit(ctx::CLContext, expr::Expr) = begin
    expr = rm_linenum!(expr)
    if haskey(visitors, expr.head)
        visitors[expr.head](ctx, expr)
    else
        error("unhandled head :$(expr.head)")
    end
end

visit(ctx, n::SymbolNode) = begin
    return CName(cname(n.name), n.typ)
end

visit(ctx, n::Symbol) = begin
    ty = get(ctx.var_types, n, Void)
    return CName(cname(n), ty)
end

visit(ctx, n::String) = begin
    return CStr(n, Ptr{Cchar})
end

visit(ctx, n::Number) = begin
    return CNum(n, typeof(n))
end

visit(ctx, n::GotoNode) = begin
    labelname = "label" * string(n.label)
    return CGoto(labelname)
end

visit(ctx, n::LabelNode) = begin
    labelname = "label" * string(n.label)
    return CLabel(labelname)
end

is_linenumber(ex::LineNumberNode) = true
is_linenumber(ex::Expr) = ex.head === :line
is_linenumber(ex) = false

visit_block(ctx, expr::Expr) = begin
    @assert expr.head == :body || expr.head == :block
    body = {}
    for ex in expr.args
        if is_linenumber(ex)
            continue
        end
        push!(body, visit(ctx, ex))
    end
    return CBlock(body)
end

visit_gotoifnot(ctx, expr::Expr) = begin
    @assert expr.head == :gotoifnot
    node = visit(ctx, expr.args[1])
    # switch comp ops because AMD compiler has a bug
    # where !(cmp) yields incorrect behavior (works on Intel)
    #if isa(node, CBinOp)
    #    if isa(node.op, CLt) ||
    ##       isa(node.op, CGt) ||
    #       isa(node.op, CLtE) ||
    #       isa(node.op, CGtE)
    #       tmp = node.left
    #       node.left = node.right
    #       node.right = tmp
    #       return CIf(node, 
    #                 CGoto("label$(expr.args[2])"),
    #                 nothing)
    #    end
    #end
    if isa(node, CUnaryOp) && isa(node.op, CNot)
        return CIf(node.operand,
                   CGoto("label$(expr.args[2])"),
                   nothing)
    else
        return CIf(CUnaryOp(CNot(), node, Bool), 
                   CGoto("label$(expr.args[2])"), 
                   nothing)
    end
end

visit_return(ctx, expr::Expr) = begin
    @assert expr.head == :return
    @assert length(expr.args) == 1
    if expr.args[1] != nothing
        node = visit(ctx, expr.args[1])
        return CReturn(node, node.ctype)
    else
        return CReturn(nothing, Void)
    end
end

visit_assign(ctx, expr::Expr) = begin
    @assert expr.head == :(=)
    target   = cname(expr.args[1])
    if !isa(expr.args[2], Expr)
        node     = visit(ctx, expr.args[2])
        ret_type = node.ctype 
        return CAssign(target, node, ret_type) 
    else
        if !(expr.args[2].typ <: Range || expr.args[2].typ <: Range1)
            node     = visit(ctx, expr.args[2])
            ret_type = node.ctype 
            return CAssign(target, node, ret_type) 
        elseif expr.args[2].typ <: Range
            ty = expr.args[2].typ
            start = visit(ctx, expr.args[2].args[2])
            step  = visit(ctx, expr.args[2].args[3]) 
            len   = visit(ctx, expr.args[2].args[4])
            return CAssignList([CAssign(CStructRef(CName(target, ty),
                                                   CName("start", Int), Int),
                                        start, 
                                        start.ctype),
                                CAssign(CStructRef(CName(target, ty),
                                                   CName("step", Int), Int),
                                        step,
                                        step.ctype),
                                CAssign(CStructRef(CName(target, ty),
                                                   CName("len", Int), Int),
                                        len,
                                        len.ctype)])
        elseif expr.args[2].typ <: Range1
            ty = expr.args[2].typ
            start = visit(ctx, expr.args[2].args[2])
            step  = CNum(1, Int)
            len   = visit(ctx, expr.args[2].args[3])
            return CAssignList([CAssign(CStructRef(CName(target, ty),
                                                   CName("start", Int), Int),
                                        start, 
                                        start.ctype),
                                CAssign(CStructRef(CName(target, ty),
                                                   CName("step", Int), Int),
                                        step,
                                        step.ctype),
                                CAssign(CStructRef(CName(target, ty),
                                                   CName("len", Int), Int),
                                        len,
                                        len.ctype)])
        else
            error("unhandled assign")
        end
    end
end

visit_colon(ctx, expr::Expr) = begin
    @assert expr.args[1] == :colon
    if length(expr.args) == 4
        start = visit(ctx, expr.args[2])
        step  = visit(ctx, expr.args[3]) 
        len   = visit(ctx, expr.args[4])
        return CArray([start, step, len], nothing, Range)
    elseif length(expr.args) == 3
        start = visit(ctx, expr.args[2])
        len   = visit(ctx, expr.args[3])
        return CArray([start, 1, len], nothing, Range)
    else
        error("colon expression should have 3 or 4 arguments")
    end
end

visit_arrayset(ctx, expr::Expr) = begin
    @assert expr.args[1] === :arrayset
    target = visit(ctx, expr.args[2])
    val = visit(ctx, expr.args[3])
    idx = visit(ctx, expr.args[4])
    ty  = target.ctype
    if isa(idx, CTypeCast)
        cast_ty = idx.ctype
        val_ty  = idx.value.ctype
        if cast_ty != Uint32
            if val_ty == Uint32
                idx = idx.value
            else
                idx = CTypeCast(idx.value, Uint32)
            end
        end
    end
    return CAssign(CSubscript(target, CIndex(idx), ty),
                   val, ty)
end

visit_arrayref(ctx, expr::Expr) = begin
    @assert expr.args[1] == :arrayref
    target = visit(ctx, expr.args[2])
    idx = visit(ctx, expr.args[3])
    ty  = array_elemtype(target.ctype)
    if isa(idx, CTypeCast)
        cast_ty = idx.ctype
        val_ty = idx.value.ctype
        if cast_ty != Uint32
            if val_ty == Uint32
                idx = idx.value
            else
                idx = CTypeCast(idx.value, Uint32)
            end
        end
    end
    return CSubscript(target, CIndex(idx), ty)
end

visit_is(ctx, expr::Expr) = begin
    @assert expr.args[1] == :(===)
    lnode = visit(ctx, expr.args[2])
    rnode = visit(ctx, expr.args[3])
    return CBinOp(lnode, CEq(), rnode, Bool)
end 
 
#TODO: this only integer indices 
visit_index(ctx, expr::Expr) = begin
    @assert expr.head == :ref
    @assert typeof(expr.args[2]) <: Integer 
    return CIndex(expr.args[2])
end

const builtin_funcs = (Symbol => String) [:pow  => "pow",
                                          :powf => "pow",
                                          :expf => "exp",
                                          :logf => "log",
                                          :log  => "log",
                                          :exp  => "exp",
                                          :sinf => "sin"] 

function call_builtin(ctx, fname, expr::Expr, ret_type::Type)
    if !haskey(builtin_funcs, fname)
        error("unknown builtin function $fname")
    end
    if fname === :powf || fname === :pow
        arg1 = visit(ctx, expr.args[5])
        arg2 = visit(ctx, expr.args[7])
        ret_type = promote_type(arg1.ctype, arg2.ctype)
        return CLRTCall("pow", [arg1, arg2], ret_type)  
    end
    if fname == :expf || fname == :exp
        arg1 = visit(ctx, expr.args[5])
        return CLRTCall("exp", [arg1,], ret_type)
    end
    if fname == :logf || fname == :log
        arg1 = visit(ctx, expr.args[5])
        return CLRTCall("log", [arg1,], ret_type)
    end
    if fname == :sinf || fname == :sin
        arg1 = visit(ctx, expr.args[5])
        return CLRTCall("sin", [arg1,], ret_type)
    end
    error("unhanded builtin $fname")
end

isfunction(sym::Symbol) = isa(eval(Main, sym), Function)

visit_callfunction(ctx, expr::Expr) = begin
    @assert isa(expr.args[1], Symbol)
    func = eval(Main, expr.args[1])
    @assert isa(func, Function)
    name = func.env.name
    typs = func.env.defs.sig
    if any(x -> !isleaftype(x), typs)
        typs = tuple([a.typ for a in expr.args[2:end]]...)
    end
    fexpr = first(code_typed(func, typs))
    fctx, ast = build_function("$name", fexpr)
    if !(isempty(fctx.funcs))
        for (n, fast) in fctx.funcs
            ctx.funcs[n] = fast
        end
    end
    ctx.funcs[name] = ast
    args = CAst[]
    for arg in expr.args[2:end]
        push!(args, visit(ctx, arg))
    end
    return CFunctionCall(cname(name), args, expr.typ)
end

visit_ccall(ctx, expr::Expr) = begin
    @assert isa(expr.args[1], TopNode)
    @assert expr.args[1].name == :ccall
    # get function name
    fname    = (expr.args[2].args[2])
    #@show fname
    #@show typeof(fname)
    #@show expr
    #@show expr.args[3]
    #@assert isa(fname, QuoteNode)
    if isa(fname, QuoteNode)
        fname = fname.value
    elseif isa(fname, String)
        fname = symbol(fname)
    else
        error("Unhandeled code path in ccall")
    end
    # get function return type
    ret_type = eval(expr.args[3])
    @assert isa(ret_type, DataType) 
    return call_builtin(ctx, fname, expr, ret_type)
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

const intrinsic_check_arithmetic = Set{Symbol}(:checked_sadd,
                                               :checked_uadd,
                                               :checked_ssub,
                                               :checked_usub,
                                               :checked_smul,
                                               :checked_umul,
                                               :nan_dom_err)
const binary_builtins = (Symbol=>CAst)[
                                    :add_int => CAdd(),
                                    :add_float => CAdd(),
                                    :sub_int   => CSub(),
                                    :sub_float => CSub(),
                                    :div_float => CDiv(),
                                    :eq_float => CEq(),
                                    :eq_int => CEq(),
                                    :le_float => CLtE(),
                                    :le_int => CLtE(),
                                    :sle_int => CLtE(),
                                    :lt_float => CLt(),
                                    :lt_int => CLt(),
                                    :ult_int => CLt(),
                                    :mul_float => CMult(),
                                    :mul_int => CMult(),
                                    :ne_float => CNotEq(),
                                    :ne_int => CNotEq(),
                                    :or_int => COr(), 
                                    :smod_int => CMod(),
                                    :lshr_int => CBitShiftRight(),
                                    :shl_int => CBitShiftLeft(),
                                    :xor_int  => CBitXor(),
                                    :and_int => CBitAnd(),
                                    ]

const unary_builtins = (Symbol=>CAst)[
                                    :not_int => CNot(),
                                    :neg_float => CUSub(),
                                    :neg_int => CUSub()]

const runtime_funcs = Set{Symbol}(:get_global_id, :get_global_size)

visit_pow(ctx, expr::Expr) = begin
    arg1 = visit(ctx, expr.args[2])
    arg2 = visit(ctx, expr.args[3])
    ret_type = promote_type(arg1.ctype, arg2.ctype)
    if arg2.ctype <: Integer
        return CLRTCall("pown", [arg1, arg2], ret_type) 
    elseif arg2.ctype <: FloatingPoint
        return CLRTCall("pow", [arg1, arg2], ret_type)
    else
        error("invalid code path in power_by_squaring")
    end
end

visit_binaryop(ctx, expr::Expr) = begin
    local op::CAst
    local arg1 = expr.args[1]
    if isa(arg1, SymbolNode) || isa(arg1, TopNode)
        op = binary_builtins[arg1.name]
    elseif isa(arg1, Symbol)
        op = binary_builtins[arg1]
    else
        error("unhandled code path in binaryop")
    end
    lnode = visit(ctx, expr.args[2])
    rnode = visit(ctx, expr.args[3])
    ret_type = expr.typ
    if ret_type === Any
        ret_type = promote_type(lnode.ctype, rnode.ctype)
    end
    return CBinOp(lnode, op, rnode, ret_type)
end

visit_unaryop(ctx, expr::Expr) = begin
    local opname::Symbol
    local arg1 = expr.args[1]
    if isa(arg1, TopNode) || isa(arg1, SymbolNode)
        opname = arg1.name
    elseif isa(arg1, Symbol)
        opname = arg1
    else
        error("unhandled code path in unaryop")
    end
    local op::CAst
    if haskey(unary_builtins, opname)
        op = unary_builtins[opname]
    elseif haskey(runtime_funcs, opname)
        op = runtime_funcs[opname]
    else
        error("unhandled binary function $opname")
    end
    node = visit(ctx, expr.args[2])
    ret_type = node.ctype 
    return CUnaryOp(op, node, ret_type)
end

visit_getfield(ctx, expr::Expr) = begin
    @assert isa(expr.args[1], GetfieldNode)
    fnode = expr.args[1]
    if fnode.value == Base.Math
        if fnode.name in intrinsic_check_arithmetic
            # pass through for checked arithmetic
            #println(expr.args)
            node = visit(ctx, expr.args[2])
            return node
        end
    end
    error("unhandled code path in getfield")
end

visit_call1(ctx, expr::Expr) = begin
    error("unhanded call1")
end

visit_call(ctx, expr::Expr) = begin
    @assert expr.head === :call
    arg1 = first(expr.args)
    
    if isa(arg1, GetfieldNode)
        return visit_getfield(ctx, expr)
    end
    
    if arg1 == :colon
        return visit_colon(ctx, expr)
    end
    if arg1 === :arrayset
        return visit_arrayset(ctx, expr)
    end
    if arg1 === :arrayref
        return visit_arrayref(ctx, expr)
    end
    if arg1 === :(===)
        return visit_is(ctx, expr)
    end
    if arg1 == :(^)
        return visit_pow(ctx, expr)
    end

    #TODO: handle runtime functions
    if arg1 in runtime_funcs
        args = CAst[]
        for arg in expr.args[2:end]
            push!(args, visit(ctx, arg))
        end
        return CFunctionCall(cname(arg1), args, Csize_t)
    end
    
    if isa(arg1, Symbol) && arg1 === :clprintf
        @show expr
        @show expr.args
        args = CAst[]
        for arg in expr.args[2:end]
            push!(args, visit(ctx, arg))
        end
        return CLRTCall("printf", args, Void)
    end

    if isa(arg1, Symbol) && isfunction(arg1)
        return visit_callfunction(ctx, expr)
    end

    # c ? b : a
    # cl select(a, b, c)
    if isa(arg1, Expr)
        if arg1.head == :call
            if arg1.args[2] === :Intrinsics
                if arg1.args[3].value === :select_value
                    cond = visit(ctx, expr.args[2])
                    arg1 = visit(ctx, expr.args[3])
                    arg2 = visit(ctx, expr.args[4])
                    ty   = promote_type(arg1.ctype, arg2.ctype)
                    if arg1.ctype != ty
                        arg1 = CTypeCast(arg1, ty)
                    end
                    if arg2.ctype != ty
                        arg2 = CTypeCast(arg2, ty)
                    end
                    return CLRTCall("select", [arg2, arg1, CTypeCast(cond, ty)], expr.typ)
                end
            end
        end
        error("unhandled code path")
    end 
    if !(isa(expr.args[1], TopNode))
        @show expr.args[1], typeof(arg1)
        @show expr.args[1].head
        @show expr.args[1].args
        @show expr.args[2]
        @show expr.args[3]
        @show expr.args[4]
        @show expr.args[5]
        error("top node error")
    end
    
    # low level ccall functions
    if arg1.name === :ccall
        return visit_ccall(ctx, expr)
     
    # unbox boxed numbers
    elseif arg1.name === :box
        # expr.args[2] is a type symbol name
        ret_type = eval(expr.args[2])
        @assert isa(ret_type, DataType)
        if !(ret_type <: CLType)
            error("invalid cast to type $ret_type")
        end
        node = visit(ctx, expr.args[3])
        #if !(promote_type(node.ctype, ret_type) == ret_type)
        #    @show expr.args[3]
        #    @show node.ctype, ret_type
        #    error("ERR promote type")
        #end

        if node.ctype === ret_type
            return node
        elseif isa(node, CNum)
            return CNum(node.val, ret_type)
        else
            return CTypeCast(node, ret_type)
        end
   
    # type assertions get translated to casts
    elseif arg1.name === :typeassert
        val = visit(ctx, expr.args[2])
        ty  = eval(expr.args[3])
        if val.ctype != nothing
            if isa(val.ctype, ty)
                # no op
                return val
            end
        end
        return CTypeCast(val, ty)
    
    elseif arg1.name === :getfield
        ty     = expr.typ
        sname  = visit(ctx, expr.args[2])
        sfield = expr.args[3]
        @assert isa(sfield, QuoteNode)
        sfield = CName(cname(sfield.value), ty) 
        return CStructRef(sname, sfield, ty)
    
    elseif arg1.name === :abs_float
        node = visit(ctx, expr.args[2])
        return CLRTCall("fabs", [node,], node.ctype)

    # pow for integer exponents >= 4  
    elseif arg1.name === :power_by_squaring
        arg1 = visit(ctx, expr.args[2])
        arg2 = visit(ctx, expr.args[3])
        ret_type = promote_type(arg1.ctype, arg2.ctype)
        if arg2.ctype <: Integer
            return CLRTCall("pown", [arg1, arg2], ret_type) 
        elseif arg2.ctype <: FloatingPoint
            return CLRTCall("pow", [arg1, arg2], ret_type)
        else
            error("invalid code path in power_by_squaring")
        end

    # cast integer to float 
    elseif arg1.name === :sitofp
        ty = expr.args[2]
        @assert ty <: Number
        n = visit(ctx, expr.args[3])
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
    
    # cast unsigned int to floating point
    elseif arg1.name === :uitofp
        ty = expr.args[2]::DataType
        node = visit(ctx, expr.args[3])
        return CTypeCast(node, ty)

    # cast signed /unsigned integers
    elseif (arg1.name === :sext_int ||
            arg1.name === :zext_int)
        node = visit(ctx, expr.args[3]) 
        return node

    # cast fp 
    elseif arg1.name === :fpext
        node = visit(ctx, expr.args[3])
        return node
    
    # less than if
    elseif arg1.name === :ltfsi64 || 
           arg1.name === :slt_int
        val = visit(ctx, expr.args[2])
        var = visit(ctx, expr.args[3])
        return CBinOp(val, CLt(), var, Cint)

    # truncated fp casting
    elseif arg1.name === :fptrunc
        ret_type =  eval(expr.args[2])
        @assert isa(ret_type, DataType)
        node = visit(ctx, expr.args[3])
        if isa(node, CNum)
            return CNum(node.val, ret_type)
        else
            return CTypeCast(node, ret_type)
        end

    elseif arg1.name === :trunc_int
        ret_type = eval(expr.args[2])
        @assert isa(ret_type, DataType)
        node = visit(ctx, expr.args[3])
        if isa(node, CNum)
            return CNum(node.val, ret_type)
        else
            return CTypeCast(node, ret_type)
        end

    elseif arg1.name === :checked_ssub
        lnode = visit(ctx, expr.args[2])
        rnode = visit(ctx, expr.args[3])
        ret_type = promote_type(lnode.ctype, rnode.ctype)
        return CBinOp(lnode, CSub(), rnode, ret_type)

    elseif arg1.name === :checked_sadd
        lnode = visit(ctx, expr.args[2])
        rnode = visit(ctx, expr.args[3])
        ret_type = promote_type(lnode.ctype, rnode.ctype)
        return CBinOp(lnode, CAdd(), rnode, ret_type)

    # binary operations
    elseif haskey(binary_builtins, arg1.name)
        visit_binaryop(ctx, expr)
      
    # unary operations
    elseif haskey(unary_builtins, arg1.name)
        visit_unaryop(ctx, expr)
    else
        @show expr
        error("unhandled call function :$(arg1)")
    end
end

visit_lambda(ctx, node::Expr) = begin
    error("visit_lambda unimplemented")
end

function build_function(name::String, expr::Expr; iskernel=false)
    @assert expr.head === :lambda
    ctx = CLContext()
    
    # parse variable declarations
    ctx.func_args = copy(expr.args[1])
    ctx.local_vars = Set{Symbol}(expr.args[2][1]...)
    for var in expr.args[2][2]
        ctx.var_types[var[1]] = var[2]
    end

    # parse args
    args = CAst[]
    for arg in ctx.func_args
        ty = ctx.var_types[arg]
        if ty <: Number 
            push!(args, CTypeDecl(cname(arg), ty))
        elseif ty <: Ptr
            push!(args, CPtrDecl(cname(arg), ty))
        elseif ty <: Array
            T = array_elemtype(ty)
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
    for var in ctx.local_vars
        ty = ctx.var_types[var]
        if ty <: Number
            push!(vars, CVarDecl(cname(var), ty))
        elseif ty <: Ptr
            push!(vars, CPtrDecl(cname(var), ty))
        elseif ty <: Array
            #TODO: arrays need to be fixed size
            push!(vars, CArrayDecl(cname(var), ty))
        elseif ty <: NTuple
            push!(vars, CVarDecl(cname(var), ty))
        elseif ty <: Range
            push!(vars, CVarDecl(cname(var), ty))
        elseif ty <: Range1
            push!(vars, CVarDecl(cname(var), ty))
        else
            @show name, var, ty
            error("unknown code path in visit_lambda loc vars")
        end
    end

    # parse body
    blocknode = visit(ctx, expr.args[end])

    # prepend variable declarations in body
    prepend!(blocknode.body, reverse!(vars))
    
    # return type
    local ret_type::Type
    if isa(blocknode.body[end], CReturn)
        ret_type = blocknode.body[end].ctype
        if ret_type <: Array && ret_type != None
            T = array_elemtype(ret_type)
            ret_type = Ptr{T}
        end
    else
        ret_type = Ptr{Void}
    end
    if iskernel
        return (ctx, CLKernelDef(name, args, blocknode, ret_type))
    else
        return (ctx, CFunctionDef(name, args, blocknode, ret_type))
    end
end

build_kernel(name::String, expr::Expr) = build_function(name, expr, iskernel=true)

const visitors = (Symbol=>Function)[:lambda => visit_lambda,
                                    :block  => visit_block,
                                    :body   => visit_block,
                                    :return => visit_return,
                                    :(=)    => visit_assign,
                                    :ref    => visit_index,
                                    :call   => visit_call,
                                    :call1  => visit_call1,
                                    :gotoifnot => visit_gotoifnot,]
end
