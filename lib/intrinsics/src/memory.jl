# local memory

# get a pointer to local memory, with known (static) or zero length (dynamic)
@generated function emit_localmemory(::Type{T}, ::Val{len}=Val(0)) where {T,len}
    Context() do ctx
        # XXX: as long as LLVMPtr is emitted as i8*, it doesn't make sense to type the GV
        eltyp = convert(LLVMType, LLVM.Int8Type())
        T_ptr = convert(LLVMType, LLVMPtr{T,AS.Workgroup})

        # create a function
        llvm_f, _ = create_function(T_ptr)

        # determine the array size: an array of a bits union stores a selector byte per
        # element after the values
        sz = len * sizeof(T)
        Base.isbitsunion(T) && (sz += len)

        # create the global variable
        mod = LLVM.parent(llvm_f)
        gv_typ = LLVM.ArrayType(eltyp, sz)
        gv = GlobalVariable(mod, gv_typ, "local_memory", AS.Workgroup)
        if len > 0
            linkage!(gv, LLVM.API.LLVMInternalLinkage)
            initializer!(gv, null(gv_typ))
        end
        # TODO: Make the alignment configurable
        align = 1
        for typ in Base.uniontypes(T)
            typ.layout != C_NULL && (align = max(align, Base.datatype_alignment(typ)))
        end
        alignment!(gv, align)

        # generate IR
        IRBuilder() do builder
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            ptr = gep!(builder, gv_typ, gv, [ConstantInt(0), ConstantInt(0)])

            untyped_ptr = bitcast!(builder, ptr, T_ptr)

            ret!(builder, untyped_ptr)
        end

        call_function(llvm_f, LLVMPtr{T,AS.Workgroup})
    end
end
