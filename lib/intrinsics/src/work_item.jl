# Work-Item Functions
#
# https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Env.html#_built_in_variables

# NOTE: these functions now unsafely truncate to Int to avoid top bit checks.
#       we should probably use range metadata instead.

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
    gvar_name = Symbol("@__spirv_$(spirv_name)")
    width = sizeof(julia_type) * 8
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
                """, "entry"), $julia_type, Tuple{}) % Int + $offset
    end
end

# 3D values
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
    gvar_name = Symbol("@__spirv_$(spirv_name)")
    width = Int === Int64 ? 64 : 32
    @eval begin
        export $julia_name
        @device_function $julia_name(dimindx::Integer=1u32) =
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
