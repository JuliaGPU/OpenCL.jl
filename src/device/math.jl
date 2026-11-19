# `@fastmath` versions of math functions
#
# SPIRVIntrinsics maps the single-precision `@fastmath` `sin`, `cos` and `tan` to the OpenCL
# `native_*` builtins. spirv2clc, which translates the SPIR-V for the OpenCL C source path,
# doesn't support those, so use the accurate functions there.

for f in (:sin, :cos, :tan)
    native = Symbol(:native_, f)
    @eval @device_override Base.FastMath.$(Symbol(f, :_fast))(x::Float32) =
        has_feature(:native_math) ? $native(x) : $f(x)
end

@device_override Base.FastMath.sincos_fast(x::Float32) =
    has_feature(:native_math) ? (native_sin(x), native_cos(x)) : sincos(x)
