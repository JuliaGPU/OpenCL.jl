module KernelAbstractionsExt

using OpenCL
using OpenCL: @device_override, method_table

import KernelAbstractions as KA

import Adapt

Adapt.adapt_storage(::KA.CPU, a::CLArray) = convert(Array, a)

# `@Const` applies `constify` inside the kernel, where arguments have already been
# converted to device arrays, so the rule has to be registered for `CLDeviceArray`
# rather than for `CLArray`.
Adapt.adapt_storage(::KA.ConstAdaptor, a::CLDeviceArray) = Base.Experimental.Const(a)

## scratch memory

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    KA.PrivateArray{T}(undef, Val(Dims), Val(OpenCL.AS.Function))
end

end
