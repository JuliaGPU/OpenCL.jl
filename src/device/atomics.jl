## feature-gated floating-point atomics

# Since SPIRVIntrinsics 1.4, the native and fallback helpers are the same UnsafeAtomics
# operation, which GPUCompiler selects to an EXT instruction or expands to a compare-and-swap
# loop, so both branches compile to the same code. They remain for older SPIRVIntrinsics.
for (T, add_feature, min_max_feature) in
        ((Float32, :fp32_atomic_add, :fp32_atomic_min_max),
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
