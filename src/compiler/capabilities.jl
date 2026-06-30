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


## compile-time feature queries (device side, folded to a constant by the optimizer)

# Load the feature bitset that `finish_module!` materializes as a module-scope constant. Once the
# constant is in place the load folds away, so `has_feature` branches resolve at compile time. The
# global uses the UniformConstant (2) storage class to stay valid SPIR-V if it ever survives.
@device_function @inline function feature_bitset()
    Base.llvmcall(
        ("""@__opencl_feature_bitset = external addrspace(2) global i64
            define i64 @entry() #0 {
                %v = load i64, i64 addrspace(2)* @__opencl_feature_bitset
                ret i64 %v
            }
            attributes #0 = { alwaysinline }
        """, "entry"), UInt64, Tuple{})
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
