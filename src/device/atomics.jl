## feature-gated floating-point atomics

# SPIRVIntrinsics defaults the floating-point atomics to a compare-and-swap loop; on devices
# that advertise native floating-point atomics (cl_ext_float_atomics), select the EXT
# instructions instead. `has_feature` folds at compile time, so only the chosen implementation
# is emitted, and `_compiler_config` only allows the corresponding SPIR-V extensions on
# devices that support them.
for (T, add_feature, min_max_feature) in
        ((Float16, :fp16_atomic_add, :fp16_atomic_min_max),
         (Float32, :fp32_atomic_add, :fp32_atomic_min_max),
         (Float64, :fp64_atomic_add, :fp64_atomic_min_max)),
    as in (AS.Workgroup, AS.CrossWorkgroup)
@eval begin

@device_override SPIRVIntrinsics.atomic_add!(p::LLVMPtr{$T,$as}, val::$T) =
    has_feature($(QuoteNode(add_feature))) ? atomic_add_native!(p, val) :
                                             atomic_add_fallback!(p, val)

@device_override SPIRVIntrinsics.atomic_sub!(p::LLVMPtr{$T,$as}, val::$T) =
    has_feature($(QuoteNode(add_feature))) ? atomic_sub_native!(p, val) :
                                             atomic_sub_fallback!(p, val)

@device_override SPIRVIntrinsics.atomic_min!(p::LLVMPtr{$T,$as}, val::$T) =
    has_feature($(QuoteNode(min_max_feature))) ? atomic_min_native!(p, val) :
                                                 atomic_min_fallback!(p, val)

@device_override SPIRVIntrinsics.atomic_max!(p::LLVMPtr{$T,$as}, val::$T) =
    has_feature($(QuoteNode(min_max_feature))) ? atomic_max_native!(p, val) :
                                                 atomic_max_fallback!(p, val)

end
end
