module KernelAbstractionsExt

using OpenCL

import KernelAbstractions as KA

import Adapt

Adapt.adapt_storage(::KA.CPU, a::CLArray) = convert(Array, a)

# `@Const` applies `constify` inside the kernel, where arguments have already been
# converted to device arrays, so the rule has to be registered for `CLDeviceArray`
# rather than for `CLArray`.
Adapt.adapt_storage(::KA.ConstAdaptor, a::CLDeviceArray) = Base.Experimental.Const(a)

end
