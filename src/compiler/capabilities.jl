## Optional device features ("aspects")
#
# A flat set of optional capabilities, like SYCL 2020 aspects. A single OpenCL C version number
# isn't enough: NVIDIA and pocl both report OpenCL C 1.2 but differ on subgroups. Each `FEATURES`
# entry is a `name` and a host-side `detect` query; a device profile is a `FeatureSet` bitset
# indexed by position, so adding a feature just claims the next bit (no GPUCompiler release needed).
#
# We don't validate a kernel's feature use against the device; the driver rejects incompatible
# SPIR-V/OpenCL C itself. Kernels select features at compile time with `has_feature` and supply
# their own fallback (see the device RNG).

struct Feature
    name::Symbol
    detect::Function
end

# OpenCL 3.0: array of cl_name_version {cl_version (4 bytes); char name[64]}. Returns [] on
# devices that don't support the query (e.g. OpenCL 1.2).
function opencl_c_features(dev::cl.Device)::Vector{String}
    CL_DEVICE_OPENCL_C_FEATURES = 0x106f
    try
        sz = Ref{Csize_t}(0)
        cl.clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_FEATURES, 0, C_NULL, sz)
        sz[] == 0 && return String[]
        buf = Vector{UInt8}(undef, sz[])
        cl.clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_FEATURES, sz[], buf, C_NULL)
        entry = 68  # sizeof(cl_name_version)
        feats = String[]
        for off in 0:entry:(length(buf) - entry)
            namebytes = @view buf[(off + 5):(off + entry)]   # skip the 4-byte version
            nul = findfirst(==(0x00), namebytes)
            name = String(namebytes[1:(nul === nothing ? length(namebytes) : nul - 1)])
            isempty(name) || push!(feats, name)
        end
        return feats
    catch
        return String[]
    end
end

has_opencl_c_feature(dev, feat) = feat in opencl_c_features(dev)

# cl_ext_float_atomics: the floating-point atomic capabilities of one precision, a bitfield
# (cl_device_fp_atomic_capabilities_ext) with separate global- and local-memory bits. Zero if
# the device doesn't support the extension.
function fp_atomic_capabilities(dev::cl.Device, query)::UInt64
    "cl_ext_float_atomics" in dev.extensions || return 0
    try
        supported = Ref{UInt64}(0)
        cl.clGetDeviceInfo(dev, query, sizeof(UInt64), supported, C_NULL)
        return supported[]
    catch
        return 0
    end
end

# the `has_feature` bits apply to both address spaces, so require both bits
has_fp_atomics(dev::cl.Device, query, caps) = fp_atomic_capabilities(dev, query) & caps == caps

# OpenCL 3.0: CL_DEVICE_OPENCL_C_ALL_VERSIONS lists every OpenCL C version the device accepts as
# an array of cl_name_version {cl_version (4 bytes); char name[64]}. This is the query to trust:
# the legacy CL_DEVICE_OPENCL_C_VERSION string reports "1.2" on both NVIDIA and pocl even though
# pocl accepts up to 3.0. Returns the highest version, falling back to the legacy string (or 1.2)
# on devices that don't support the 3.0 query.
function max_opencl_c_version(dev::cl.Device)::VersionNumber
    CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066
    try
        sz = Ref{Csize_t}(0)
        cl.clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, C_NULL, sz)
        if sz[] != 0
            buf = Vector{UInt8}(undef, sz[])
            cl.clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_ALL_VERSIONS, sz[], buf, C_NULL)
            entry = 68  # sizeof(cl_name_version)
            best = v"0"
            for off in 0:entry:(length(buf) - entry)
                ver = reinterpret(UInt32, buf[(off + 1):(off + 4)])[1]  # cl_version bitfield
                vn = VersionNumber(ver >> 22, (ver >> 12) & 0x3ff)      # major[31:22], minor[21:12]
                vn > best && (best = vn)
            end
            best > v"0" && return best
        end
    catch
    end
    return legacy_opencl_c_version(dev)
end

# Pre-3.0 fallback: parse the "OpenCL C <major>.<minor> <vendor>" string.
function legacy_opencl_c_version(dev::cl.Device)::VersionNumber
    CL_DEVICE_OPENCL_C_VERSION = 0x103d
    try
        sz = Ref{Csize_t}(0)
        cl.clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, 0, C_NULL, sz)
        buf = Vector{UInt8}(undef, sz[])
        cl.clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, sz[], buf, C_NULL)
        str = String(buf[1:(end - 1)])  # drop the trailing NUL
        m = match(r"OpenCL C (\d+)\.(\d+)", str)
        m !== nothing && return VersionNumber(parse(Int, m[1]), parse(Int, m[2]))
    catch
    end
    return v"1.2"  # conservative default
end

const FEATURES = Feature[
    Feature(:fp16, dev -> "cl_khr_fp16" in dev.extensions),
    Feature(:fp64, dev -> "cl_khr_fp64" in dev.extensions),
    Feature(:int64_atomics, dev -> "cl_khr_int64_base_atomics" in dev.extensions),
    Feature(:subgroups, cl.sub_groups_supported),
    Feature(:generic_address_space,
            dev -> has_opencl_c_feature(dev, "__opencl_c_generic_address_space")),
    # What the device reports for floating-point atomics in both global and local memory.
    # These don't decide whether kernels use native instructions: that follows from the
    # compiler target's `SPIRVAtomics` (see `device_atomics`), which can differ per address
    # space and is restricted further on the OpenCL C source path.
    Feature(:fp32_atomic_add,
            dev -> has_fp_atomics(dev, cl.CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT,
                                  cl.CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                  cl.CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)),
    Feature(:fp32_atomic_min_max,
            dev -> has_fp_atomics(dev, cl.CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT,
                                  cl.CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                  cl.CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)),
    Feature(:fp64_atomic_add,
            dev -> has_fp_atomics(dev, cl.CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT,
                                  cl.CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                  cl.CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)),
    Feature(:fp64_atomic_min_max,
            dev -> has_fp_atomics(dev, cl.CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT,
                                  cl.CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                  cl.CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)),
]

const FeatureSet = UInt64
@assert length(FEATURES) <= 64

function feature_index(name::Symbol)
    for (i, f) in enumerate(FEATURES)
        f.name === name && return i
    end
    throw(ArgumentError("unknown OpenCL feature $name"))
end

feature_bit(name::Symbol) = one(FeatureSet) << (feature_index(name) - 1)
feature_supported(fs::FeatureSet, name::Symbol) = (fs & feature_bit(name)) != 0

"""
    device_features(dev::cl.Device) -> FeatureSet

The set of optional features the device supports.
"""
function device_features(dev::cl.Device)::FeatureSet
    fs = zero(FeatureSet)
    for (i, f) in enumerate(FEATURES)
        f.detect(dev) && (fs |= one(FeatureSet) << (i - 1))
    end
    return fs
end

feature_supported(dev::cl.Device, name::Symbol) = feature_supported(device_features(dev), name)


## atomic operations

# The OpenCL C version that programs on the source path are compiled for: the device's highest.
source_opencl_c_version(dev::cl.Device) = max_opencl_c_version(dev)

# Whether spirv2clc's translation of single- and double-precision atomic addition compiles as
# OpenCL C `version`, given the device's OpenCL C `features`. It uses the C11 `atomic_fetch_add`
# without explicit ordering and scope, i.e., seq_cst at device scope, which OpenCL C 2.0
# introduced and 3.0 made optional.
function source_fadd_supported(version::VersionNumber, features)
    version < v"2.0" && return false
    version < v"3.0" && return true
    return "__opencl_c_atomic_order_seq_cst" in features &&
           "__opencl_c_atomic_scope_device" in features
end

"""
    device_atomics(dev::cl.Device; source::Bool=false) -> SPIRVAtomics

The atomic capabilities of `dev` and its toolchain, i.e., the atomic operations GPUCompiler
may select SPIR-V instructions for; it implements the others with compare-and-swap loops.
`source` restricts them to what the OpenCL C source path (spirv2clc) can translate.

This is the default for the `atomics` compiler keyword (e.g., `@opencl atomics=...`), which
overrides the atomic capabilities GPUCompiler may select directly: it replaces the whole
device-derived `OpenCL.SPIRVAtomics` (from GPUCompiler). Enabling capabilities the device or
toolchain doesn't support can make compilation fail, or terminate the driver's compiler.
Disabling them relies on integer compare-and-swap for the fallback.
"""
function device_atomics(dev::cl.Device; source::Bool=false)
    fp16 = "cl_khr_fp16" in dev.extensions
    fp64 = "cl_khr_fp64" in dev.extensions
    int64 = "cl_khr_int64_base_atomics" in dev.extensions &&
            "cl_khr_int64_extended_atomics" in dev.extensions

    half = fp16 ? fp_atomic_capabilities(dev, cl.CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT) : 0
    single = fp_atomic_capabilities(dev, cl.CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT)
    double = fp64 ? fp_atomic_capabilities(dev, cl.CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT) : 0

    if source
        # spirv2clc doesn't translate half-precision atomic addition
        half = 0
        if !source_fadd_supported(source_opencl_c_version(dev), opencl_c_features(dev))
            single = double = 0
        end
    end

    global_add(caps) = caps & cl.CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT != 0
    local_add(caps) = caps & cl.CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT != 0
    return SPIRVAtomics(; int64,
                        fadd_f16_global = global_add(half), fadd_f16_local = local_add(half),
                        fadd_f32_global = global_add(single), fadd_f32_local = local_add(single),
                        fadd_f64_global = global_add(double), fadd_f64_local = local_add(double))
end


## compile-time feature queries (device side, folded to a constant by the optimizer)

# Load the feature bitset that `finish_module!` materializes as a module-scope constant. Once the
# constant is in place the load folds away, so `has_feature` branches resolve at compile time. The
# global uses the UniformConstant (2) storage class to stay valid SPIR-V if it ever survives.
@device_function @inline feature_bitset() = _feature_bitset()
@llvmgenerated builder function _feature_bitset()::UInt64
    T = LLVM.Int64Type()
    gv = GlobalVariable(current_module(builder), T, "__opencl_feature_bitset",
                        AS.UniformConstant)
    load!(builder, T, gv)
end

export has_feature

"""
    has_feature(name::Symbol) -> Bool

Compile-time query (device side): does the kernel's target device support optional feature `name`
(see `FEATURES`)? Folds to a constant, so `if has_feature(:subgroups) … else … end` keeps only the
live branch. Host-side, use `feature_supported(dev, name)`.
"""
@inline has_feature(name::Symbol) = has_feature(Val(name))
@generated function has_feature(::Val{name}) where {name}
    bit = feature_bit(name)   # resolved at compile time from the registry
    return :((feature_bitset() & $bit) != zero(FeatureSet))
end
