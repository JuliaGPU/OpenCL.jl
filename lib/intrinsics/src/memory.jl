# local memory

# get a pointer to local memory, with known (static) or zero length (dynamic)
@llvmgenerated builder function emit_localmemory(::Type{T},
                                                 ::Val{len}=Val(0))::LLVMPtr{T,AS.Workgroup} where {T,len}
    # XXX: as long as LLVMPtr is emitted as i8*, it doesn't make sense to type the GV
    eltyp = LLVM.Int8Type()
    T_ptr = convert(LLVMType, LLVMPtr{T,AS.Workgroup})

    # determine the array size: an array of a bits union stores a selector byte per
    # element after the values
    sz = len * sizeof(T)
    Base.isbitsunion(T) && (sz += len)

    # create the global variable
    gv_typ = LLVM.ArrayType(eltyp, sz)
    gv = GlobalVariable(current_module(builder), gv_typ, "local_memory", AS.Workgroup)
    if len > 0
        gv.linkage = LLVM.Linkage.Internal
        gv.initializer = null(gv_typ)
    end
    # TODO: Make the alignment configurable
    align = 1
    for typ in Base.uniontypes(T)
        typ.layout != C_NULL && (align = max(align, Base.datatype_alignment(typ)))
    end
    gv.alignment = align

    ptr = gep!(builder, gv_typ, gv, [ConstantInt(0), ConstantInt(0)])
    bitcast!(builder, ptr, T_ptr)
end
