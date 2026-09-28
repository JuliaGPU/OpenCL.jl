# local memory

# get a pointer to local memory, with known (static) or zero length (dynamic)
@llvmgenerated builder function emit_localmemory(::Type{T},
                                                 ::Val{len}=Val(0))::LLVMPtr{T,AS.Workgroup} where {T,len}
    # XXX: as long as LLVMPtr is emitted as i8*, it doesn't make sense to type the GV
    eltyp = LLVM.Int8Type()
    T_ptr = convert(LLVMType, LLVMPtr{T,AS.Workgroup})

    # create the global variable
    gv_typ = LLVM.ArrayType(eltyp, len * sizeof(T))
    gv = GlobalVariable(current_module(builder), gv_typ, "local_memory", AS.Workgroup)
    if len > 0
        linkage!(gv, LLVM.API.LLVMInternalLinkage)
        initializer!(gv, null(gv_typ))
    end
    # TODO: Make the alignment configurable
    alignment!(gv, Base.datatype_alignment(T))

    ptr = gep!(builder, gv_typ, gv, [ConstantInt(0), ConstantInt(0)])
    bitcast!(builder, ptr, T_ptr)
end
