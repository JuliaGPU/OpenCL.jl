# Work-Item Functions

# NOTE: these functions now unsafely truncate to Int to avoid top bit checks.
#       we should probably use range metadata instead.

# 1D values
for (julia_name, (spirv_name, offset)) in [
        # indices
        :get_global_linear_id           => (:BuiltInGlobalLinearId, 1u32),
        :get_local_linear_id            => (:BuiltInLocalInvocationIndex, 1u32),
        :get_sub_group_id               => (:BuiltInSubgroupId, 1u32),
        :get_sub_group_local_id         => (:BuiltInSubgroupLocalInvocationId, 1u32),
        # sizes
        :get_work_dim                   => (:BuiltInWorkDim, 0u32),
        :get_sub_group_size             => (:BuiltInSubgroupSize, 0u32),
        :get_max_sub_group_size         => (:BuiltInSubgroupMaxSize, 0u32),
        :get_num_sub_groups             => (:BuiltInNumSubgroups, 0u32),
        :get_enqueued_num_sub_groups    => (:BuiltInNumEnqueuedSubgroups, 0u32)]
    gvar_name = Symbol("@__spirv_$(spirv_name)")
    width = Int === Int64 ? 64 : 32
    @eval begin
        export $julia_name
        @device_function $julia_name() =
            Base.llvmcall(
                $("""$gvar_name = external addrspace($(AS.Input)) global i$(width)
                     define i$(width) @entry() #0 {
                         %val = load i$(width), i$(width) addrspace($(AS.Input))* $gvar_name
                         ret i$(width) %val
                     }
                     attributes #0 = { alwaysinline }
                """, "entry"), UInt, Tuple{}) % Int + $offset
    end
end

# 3D values
for (julia_name, (spirv_name, offset)) in [
        # indices
        :get_global_id              => (:BuiltInGlobalInvocationId, 1u32),
        :get_global_offset          => (:BuiltInGlobalOffset, 1u32),
        :get_local_id               => (:BuiltInLocalInvocationId, 1u32),
        :get_group_id               => (:BuiltInWorkgroupId, 1u32),
        # sizes
        :get_global_size            => (:BuiltInGlobalSize, 0u32),
        :get_local_size             => (:BuiltInWorkgroupSize, 0u32),
        :get_enqueued_local_size    => (:BuiltInEnqueuedWorkgroupSize, 0u32),
        :get_num_groups             => (:BuiltInNumWorkgroups, 0u32)]
    gvar_name = Symbol("@__spirv_$(spirv_name)")
    width = Int === Int64 ? 64 : 32
    @eval begin
        export $julia_name
        @device_function $julia_name(dimindx::Integer=1) =
            Base.llvmcall(
                $("""$gvar_name = external addrspace($(AS.Input)) global <3 x i$(width)>
                     define i$(width) @entry(i$(width) %idx) #0 {
                         %val = load <3 x i$(width)>, <3 x i$(width)> addrspace($(AS.Input))* $gvar_name
                         %element = extractelement <3 x i$(width)> %val, i$(width) %idx
                         ret i$(width) %element
                     }
                     attributes #0 = { alwaysinline }
                """, "entry"), UInt, Tuple{UInt}, UInt(dimindx - 1u32)) % Int + $offset
    end
end
