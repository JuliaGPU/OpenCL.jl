export sub_group_shuffle, sub_group_shuffle_xor

# Shuffles from `cl_khr_subgroup_shuffle`. A lane that is out of range (e.g. `i < 1`) gives an
# undefined value, as with the OpenCL C built-ins, so the lane is passed modulo `UInt32`
# rather than converted with a check.

const gentypes = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float16, Float32, Float64]

for gentype in gentypes
    @eval begin
        @device_function sub_group_shuffle(x::$gentype, i::Integer) =
            @builtin_ccall("__spirv_GroupNonUniformShuffle", $gentype,
                           (UInt32, $gentype, UInt32),
                           UInt32(Scope.Subgroup), x, (i - 1) % UInt32, convergent = true)
        @device_function sub_group_shuffle_xor(x::$gentype, mask::Integer) =
            @builtin_ccall("__spirv_GroupNonUniformShuffleXor", $gentype,
                           (UInt32, $gentype, UInt32),
                           UInt32(Scope.Subgroup), x, mask % UInt32, convergent = true)
    end
end
