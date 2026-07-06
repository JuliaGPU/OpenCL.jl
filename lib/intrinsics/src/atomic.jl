# Atomic Functions

# Integer atomics are emitted as SPIR-V wrapper builtins, so the LLVM SPIR-V
# backend lowers them to OpAtomic* instructions directly.

const atomic_float_types = [Float32, Float64]
const atomic_integer_types = [UInt32, Int32, UInt64, Int64]
const atomic_memory_types = [AS.Workgroup, AS.CrossWorkgroup]

const atomic_scope = Scope.Workgroup

atomic_memory_semantics(::Val{AS.Workgroup}) = MemorySemantics.WorkgroupMemory
atomic_memory_semantics(::Val{AS.CrossWorkgroup}) = MemorySemantics.CrossWorkgroupMemory


# generically typed

for gentype in atomic_integer_types, as in atomic_memory_types
    atomic_min_intrinsic = gentype <: Signed ? "__spirv_AtomicSMin" : "__spirv_AtomicUMin"
    atomic_max_intrinsic = gentype <: Signed ? "__spirv_AtomicSMax" : "__spirv_AtomicUMax"
@eval begin

@device_function atomic_add!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicIAdd", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_sub!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicISub", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_inc!(p::LLVMPtr{$gentype,$as}) =
    atomic_add!(p, one($gentype))

@device_function atomic_dec!(p::LLVMPtr{$gentype,$as}) =
    atomic_sub!(p, one($gentype))

@device_function atomic_min!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall($atomic_min_intrinsic, $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_max!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall($atomic_max_intrinsic, $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_and!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicAnd", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_or!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicOr", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_xor!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicXor", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_xchg!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicExchange", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_cmpxchg!(p::LLVMPtr{$gentype,$as}, cmp::$gentype, val::$gentype) =
    @builtin_ccall("__spirv_AtomicCompareExchange", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, UInt32, $gentype, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))),
                   UInt32(atomic_memory_semantics(Val($as))), val, cmp)
end
end


# floating-point atomics.
#
# Native floating-point atomic add and min/max come from the SPV_EXT_shader_atomic_float_add
# and SPV_EXT_shader_atomic_float_min_max extensions, emitted as SPIR-V wrapper builtins like
# the integer atomics above (OpenCL-style builtins with float arguments are not recognized by
# the Khronos translator, and only atomic_add by the LLVM SPIR-V backend). Extension support
# is an optional device capability (cl_ext_float_atomics), so the generic functions default
# to a compare-and-swap loop (correct on any device with integer cmpxchg); back-ends that can
# query the device override them to select the native version at compile time (see OpenCL.jl's
# `has_feature`).
for gentype in atomic_float_types, as in atomic_memory_types
    bits = gentype == Float32 ? UInt32 : UInt64
@eval begin

@device_function atomic_add_native!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicFAddEXT", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

# SPIR-V has no atomic float subtraction; add the negated value
@device_function atomic_sub_native!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    atomic_add_native!(p, -val)

@device_function atomic_min_native!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicFMinEXT", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

@device_function atomic_max_native!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    @builtin_ccall("__spirv_AtomicFMaxEXT", $gentype,
                   (LLVMPtr{$gentype,$as}, UInt32, UInt32, $gentype),
                   p, UInt32(atomic_scope),
                   UInt32(atomic_memory_semantics(Val($as))), val)

end

# the loops compare raw bits, not values: `==` on floats would spin forever on a stored NaN,
# and would treat a failed exchange as successful when -0.0 compares equal to 0.0.
for (op, expr) in [:add => :(old + val), :min => :(min(old, val)), :max => :(max(old, val))]
    fallback = Symbol("atomic_$(op)_fallback!")
    fn = Symbol("atomic_$(op)!")
@eval begin

@device_function @inline function $fallback(p::LLVMPtr{$gentype,$as}, val::$gentype)
    ip = reinterpret(LLVMPtr{$bits,$as}, p)
    cmp = Base.unsafe_load(ip, 1)
    while true
        old = reinterpret($gentype, cmp)
        new = reinterpret($bits, $expr)
        seen = atomic_cmpxchg!(ip, cmp, new)
        seen == cmp && return old
        cmp = seen
    end
end

@device_function $fn(p::LLVMPtr{$gentype,$as}, val::$gentype) = $fallback(p, val)

end
end

@eval begin

@device_function atomic_sub_fallback!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    atomic_add_fallback!(p, -val)

@device_function atomic_sub!(p::LLVMPtr{$gentype,$as}, val::$gentype) =
    atomic_sub_fallback!(p, val)

end
end


# specifically typed

for as in atomic_memory_types
@eval begin

# There is native support for atomic_xchg on Float32, but not for Float64,
# so we always reinterpret for consistency.
@device_function atomic_xchg!(p::LLVMPtr{Float32,$as}, val::Float32) =
    reinterpret(Float32, atomic_xchg!(reinterpret(LLVMPtr{UInt32,$as}, p),
                                      reinterpret(UInt32, val)))
@device_function atomic_xchg!(p::LLVMPtr{Float64,$as}, val::Float64) =
    reinterpret(Float64, atomic_xchg!(reinterpret(LLVMPtr{UInt64,$as}, p),
                                      reinterpret(UInt64, val)))

@device_function atomic_cmpxchg!(p::LLVMPtr{Float32,$as}, cmp::Float32, val::Float32) =
    reinterpret(Float32, atomic_cmpxchg!(reinterpret(LLVMPtr{UInt32,$as}, p),
                                         reinterpret(UInt32, cmp),
                                         reinterpret(UInt32, val)))
@device_function atomic_cmpxchg!(p::LLVMPtr{Float64,$as}, cmp::Float64, val::Float64) =
    reinterpret(Float64, atomic_cmpxchg!(reinterpret(LLVMPtr{UInt64,$as}, p),
                                         reinterpret(UInt64, cmp),
                                         reinterpret(UInt64, val)))

end
end



# documentation

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old + val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_add!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old - val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_sub!

"""
Swaps the old value stored at location `p` with new value given by `val`.
Returns old value.
"""
atomic_xchg!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute (`old` + 1) and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_inc!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute (`old` - 1) and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_dec!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `(old == cmp) ? val : old` and store result at location pointed by `p`.
The function returns `old`.
"""
atomic_cmpxchg!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `min(old, val)` and store minimum value at location pointed by `p`. The
function returns `old`.
"""
atomic_min!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `max(old, val)` and store maximum value at location pointed by `p`. The
function returns `old`.
"""
atomic_max

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old & val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_and!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old | val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_or!

"""
Read the 32-bit value (referred to as `old`) stored at location pointed by `p`.
Compute `old ^ val` and store result at location pointed by `p`. The function
returns `old`.
"""
atomic_xor!



#
# High-level interface
#

# prototype of a high-level interface for performing atomic operations on arrays
#
# this design could be generalized by having atomic {field,array}{set,ref} accessors, as
# well as acquire/release operations to implement the fallback functionality where any
# operation can be applied atomically.

const inplace_ops = Dict(
    :(+=) => :(+),
    :(-=) => :(-),
    :(*=) => :(*),
    :(/=) => :(/),
    :(÷=) => :(÷),
    :(&=) => :(&),
    :(|=) => :(|),
    :(⊻=) => :(⊻),
)

struct AtomicError <: Exception
    msg::AbstractString
end

Base.showerror(io::IO, err::AtomicError) =
    print(io, "AtomicError: ", err.msg)

"""
    @atomic a[I] = op(a[I], val)
    @atomic a[I] ...= val

Atomically perform a sequence of operations that loads an array element `a[I]`, performs the
operation `op` on that value and a second value `val`, and writes the result back to the
array. This sequence can be written out as a regular assignment, in which case the same
array element should be used in the left and right hand side of the assignment, or as an
in-place application of a known operator. In both cases, the array reference should be pure
and not induce any side-effects.

!!! warn
    This interface is experimental, and might change without warning.  Use the lower-level
    `atomic_...!` functions for a stable API.
"""
macro atomic(ex)
    # decode assignment and call
    if ex.head == :(=)
        ref = ex.args[1]
        rhs = ex.args[2]
        Meta.isexpr(rhs, :call) || throw(AtomicError("right-hand side of an @atomic assignment should be a call"))
        op = rhs.args[1]
        if rhs.args[2] != ref
            throw(AtomicError("right-hand side of a non-inplace @atomic assignment should reference the left-hand side"))
        end
        val = rhs.args[3]
    elseif haskey(inplace_ops, ex.head)
        op = inplace_ops[ex.head]
        ref = ex.args[1]
        val = ex.args[2]
    else
        throw(AtomicError("unknown @atomic expression"))
    end

    # decode array expression
    Meta.isexpr(ref, :ref) || throw(AtomicError("@atomic should be applied to an array reference expression"))
    array = ref.args[1]
    indices = Expr(:tuple, ref.args[2:end]...)

    esc(quote
        $atomic_arrayset($array, $indices, $op, $val)
    end)
end

# FIXME: make this respect the indexing style
@inline atomic_arrayset(A::AbstractArray{T}, Is::Tuple, op::Function, val) where {T} =
    atomic_arrayset(A, Base._to_linear_index(A, Is...), op, convert(T, val))

# native atomics
# TODO: support inc/dec
# TODO: this depends on backend support for the corresponding SPIR-V atomic
#       operation. Floating-point arithmetic should hit the cmpxchg fallback
#       unless a caller explicitly uses a floating-point atomic extension.
for (op,impl) in [(+)      => atomic_add!,
                  (-)      => atomic_sub!,
                  (&)      => atomic_and!,
                  (|)      => atomic_or!,
                  (⊻)      => atomic_xor!,
                  Base.max => atomic_max!,
                  Base.min => atomic_min!]
    @eval @inline atomic_arrayset(A::AbstractArray{T}, I::Integer, ::typeof($op),
                                  val::T) where {T <: Union{atomic_integer_types...}} =
        $impl(pointer(A, I), val)
end

# fallback using compare-and-swap
# TODO: for 64-bit types, this depends on backend support for 64-bit cmpxchg.
function atomic_arrayset(A::AbstractArray{T}, I::Integer, op::Function, val) where {T}
    ptr = pointer(A, I)
    old = Base.unsafe_load(ptr, 1)
    while true
        cmp = old
        new = convert(T, op(old, val))
        old = atomic_cmpxchg!(ptr, cmp, new)
        (old == cmp) && return new
    end
end
