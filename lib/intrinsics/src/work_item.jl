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


# Sub-group shuffle intrinsics using a loop and @eval, matching the style of the 1D/3D value loops above
export sub_group_shuffle, sub_group_shuffle_xor

for (jltype, llvmtype, julia_type_str) in [
        (Int8,    "i8",    :Int8),
        (UInt8,   "i8",    :UInt8),
        (Int16,   "i16",   :Int16),
        (UInt16,  "i16",   :UInt16),
        (Int32,   "i32",   :Int32),
        (UInt32,  "i32",   :UInt32),
        (Int64,   "i64",   :Int64),
        (UInt64,  "i64",   :UInt64),
        (Float16, "half",  :Float16),
        (Float32, "float", :Float32),
        (Float64, "double",:Float64)
    ]
    @eval begin
        export sub_group_shuffle, sub_group_shuffle_xor
        function sub_group_shuffle(x::$jltype, idx::Integer)
            Base.llvmcall(
                $("""
                declare $llvmtype @__spirv_GroupNonUniformShuffle(i32, $llvmtype, i32)
                define $llvmtype @entry($llvmtype %val, i32 %idx) #0 {
                    %res = call $llvmtype @__spirv_GroupNonUniformShuffle(i32 3, $llvmtype %val, i32 %idx)
                    ret $llvmtype %res
                }
                attributes #0 = { alwaysinline }
                """, "entry"), $julia_type_str, Tuple{$julia_type_str, Int32}, x, Int32(idx))
        end
        function sub_group_shuffle_xor(x::$jltype, mask::Integer)
            Base.llvmcall(
                $("""
                declare $llvmtype @__spirv_GroupNonUniformShuffleXor(i32, $llvmtype, i32)
                define $llvmtype @entry($llvmtype %val, i32 %mask) #0 {
                    %res = call $llvmtype @__spirv_GroupNonUniformShuffleXor(i32 3, $llvmtype %val, i32 %mask)
                    ret $llvmtype %res
                }
                attributes #0 = { alwaysinline }
                """, "entry"), $julia_type_str, Tuple{$julia_type_str, Int32}, x, Int32(mask))
        end
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
