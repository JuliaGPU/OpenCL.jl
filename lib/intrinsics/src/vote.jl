export sub_group_any, sub_group_all, sub_group_ballot

# Sub-group votes. These call the OpenCL C built-ins, which the SPIR-V back-end lowers to
# `OpGroupAny`, `OpGroupAll` and `OpGroupNonUniformBallot`, since `@builtin_ccall` can't
# mangle the `bool` arguments of the corresponding SPIR-V wrapper builtins.

# `cl_khr_subgroups`
@device_function sub_group_any(predicate::Bool) =
    @builtin_ccall("sub_group_any", Int32, (Int32,), Int32(predicate), convergent = true) != Int32(0)

@device_function sub_group_all(predicate::Bool) =
    @builtin_ccall("sub_group_all", Int32, (Int32,), Int32(predicate), convergent = true) != Int32(0)

# `cl_khr_subgroup_ballot`: bit `i` (counting from the least significant bit of the first
# component) is set for the work-item with (0-based) sub-group local id `i`
@device_function sub_group_ballot(predicate::Bool) =
    @builtin_ccall("sub_group_ballot", NTuple{4, VecElement{UInt32}}, (Int32,), Int32(predicate), convergent = true)
