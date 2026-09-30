# Work-Item Functions
#
# https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Env.html#_built_in_variables

# NOTE: these functions now unsafely truncate to Int to avoid top bit checks.
#       we should probably use range metadata instead.

# load a built-in variable, which the SPIR-V back-end expects as an external global in the
# Input storage class
@llvmgenerated builder function builtin_variable(::Val{name}, ::Type{T})::T where {name,T}
    T_val = convert(LLVMType, T)
    gv = GlobalVariable(current_module(builder), T_val, String(name), AS.Input)
    load!(builder, T_val, gv)
end

# load a component of a built-in vector variable, by calling the function that the SPIR-V
# back-end lowers to a load and an extract of the component
@llvmgenerated builder function builtin_vector_variable(::Val{name}, idx::Int32)::UInt where {name}
    ft = LLVM.FunctionType(convert(LLVMType, UInt), [idx.value_type])
    f = LLVM.Function(current_module(builder), String(name), ft)
    push!(f.function_attributes, EnumAttribute(:nounwind))
    push!(f.function_attributes, EnumAttribute(:willreturn))
    f.memory_effects = MemoryEffects(:none)
    call!(builder, ft, f, [idx])
end

# 1D values
for (julia_name, (spirv_name, julia_type, offset)) in [
        # indices
        :get_global_linear_id           => (:BuiltInGlobalLinearId, Csize_t, 1),
        :get_local_linear_id            => (:BuiltInLocalInvocationIndex, Csize_t, 1),
        :get_sub_group_id               => (:BuiltInSubgroupId, UInt32, 1),
        :get_sub_group_local_id         => (:BuiltInSubgroupLocalInvocationId, UInt32, 1),
        # sizes
        :get_work_dim                   => (:BuiltInWorkDim, UInt32, 0),
        :get_sub_group_size             => (:BuiltInSubgroupSize, UInt32, 0),
        :get_max_sub_group_size         => (:BuiltInSubgroupMaxSize, UInt32, 0),
        :get_num_sub_groups             => (:BuiltInNumSubgroups, UInt32, 0),
        :get_enqueued_num_sub_groups    => (:BuiltInNumEnqueuedSubgroups, UInt32, 0)]
    gvar_name = Symbol("__spirv_$(spirv_name)")
    @eval begin
        export $julia_name
        @device_function $julia_name() =
            builtin_variable(Val($(QuoteNode(gvar_name))), $julia_type) % Int + $offset
    end
end

# 3D values
#
# These are called as functions, which the SPIR-V back-end lowers to a load of the built-in
# variable and an extract of the component. Emitting that load and extract ourselves lets
# InstCombine fold a truncation of the result into a load of a vector type that SPIR-V
# doesn't have (e.g. `trunc i64 to i8` into a load of `<24 x i8>`).
for (julia_name, (spirv_name, offset)) in [
        # indices
        :get_global_id              => (:BuiltInGlobalInvocationId, 1),
        :get_global_offset          => (:BuiltInGlobalOffset, 1),
        :get_local_id               => (:BuiltInLocalInvocationId, 1),
        :get_group_id               => (:BuiltInWorkgroupId, 1),
        # sizes
        :get_global_size            => (:BuiltInGlobalSize, 0),
        :get_local_size             => (:BuiltInWorkgroupSize, 0),
        :get_enqueued_local_size    => (:BuiltInEnqueuedWorkgroupSize, 0),
        :get_num_groups             => (:BuiltInNumWorkgroups, 0)]
    fname = "__spirv_$(spirv_name)"
    mangled = Symbol("_Z$(length(fname))$(fname)i")
    push!(known_intrinsics, String(mangled))
    @eval begin
        export $julia_name
        @device_function $julia_name(dimindx::Integer=1u32) =
            builtin_vector_variable(Val($(QuoteNode(mangled))), (dimindx - 1u32) % Int32) % Int + $offset
    end
end
