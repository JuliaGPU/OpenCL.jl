using SPIRVIntrinsics: @builtin_ccall, @typed_ccall, LLVMPtr, known_intrinsics
import SPIRVIntrinsics, SPIRV_LLVM_Translator_jll, GPUCompiler

# Define the types to test
integer_types = [Int32, UInt32, Int64, UInt64]
float_types = [Float32, Float64]
all_types = vcat(integer_types, float_types)

dev = OpenCL.cl.device()

# Arithmetic operations
function test_atomic_add(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] += one(T)
    return
end
function test_atomic_sub(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] -= one(T)
    return
end
# Bitwise operations
function test_atomic_and(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] &= ~(one(T) << (get_global_id() - 1))
    return
end
function test_atomic_or(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] |= one(T) << (get_global_id() - 1)
    return
end
function test_atomic_xor(counter::AbstractArray{T}) where T
    OpenCL.@atomic counter[] ⊻= one(T) << ((get_global_id() - 1) % 32)
    return
end
# Min/max operations - use low-level API directly
function test_atomic_max(counter::AbstractArray{T}) where T
    OpenCL.atomic_max!(pointer(counter), T(get_global_id()))
    return
end
function test_atomic_min(counter::AbstractArray{T}) where T
    OpenCL.atomic_min!(pointer(counter), T(get_global_id()))
    return
end
# Exchange operation - use low-level API directly
function test_atomic_xchg(counter::AbstractArray{T}) where T
    OpenCL.atomic_xchg!(pointer(counter), one(T))
    return
end
# Compare-and-swap operation - use low-level API directly
function test_atomic_cas(counter::AbstractArray{T}) where T
    OpenCL.atomic_cmpxchg!(pointer(counter), zero(T), one(T))
    return
end
# Floating-point add/sub - use low-level API directly
function float_add_kernel(counter::AbstractArray{T}, val::T) where T
    OpenCL.atomic_add!(pointer(counter), val)
    return
end
function float_sub_kernel(counter::AbstractArray{T}, val::T) where T
    OpenCL.atomic_sub!(pointer(counter), val)
    return
end

# Define atomic operations to test
atomic_operations = [
    # op, init_val, expected_val
    (test_atomic_add, 0, 1000),
    (test_atomic_sub, 1000, 0),
    (test_atomic_and, typemax(UInt64), 0),
    (test_atomic_or, 0, typemax(UInt64)),
    (test_atomic_xor, 0, typemax(UInt32) << 8),
    (test_atomic_max, 0, 1000),
    (test_atomic_min, 1000, 1),
    (test_atomic_xchg, 0, 1),
    (test_atomic_cas, 0, 1),
]
@testset "atomics" begin
@testset "$kernel_func - $T" for (kernel_func, init_val, expected_val) in atomic_operations, T in all_types
    # Skip Int64/UInt64 if not supported
    if sizeof(T) == 8 && T <: Integer && !("cl_khr_int64_extended_atomics" in dev.extensions)
        continue
    end

    # Skip Float64 if not supported
    if T == Float64 && !("cl_khr_fp64" in dev.extensions)
        continue
    end

    # Float64 atomics may fall back to 64-bit cmpxchg
    if T == Float64 && !("cl_khr_int64_base_atomics" in dev.extensions)
        continue
    end

    # Bitwise operations (only valid for integers)
    if kernel_func in [test_atomic_and, test_atomic_or, test_atomic_xor] && T <: AbstractFloat
        continue
    end

    # Min/max on integers is only supported for 32-bit types; floats use the native
    # extension or the compare-and-swap fallback
    if kernel_func in [test_atomic_min, test_atomic_max] && !(T in [Int32, UInt32] || T <: AbstractFloat)
        continue
    end

    if T <: Integer
        init_val %= T
        expected_val %= T
    end

    a = OpenCL.fill(T(init_val))
    @opencl global_size=1000 kernel_func(a)
    result_val = OpenCL.@allowscalar a[]
    @test result_val === T(expected_val)
end


@testset "float atomics ($T)" for T in [Float32, Float64]
    if T == Float64 && !("cl_khr_fp64" in dev.extensions)
        continue
    end
    if T == Float64 && !("cl_khr_int64_base_atomics" in dev.extensions)
        continue
    end

    a = OpenCL.zeros(T)
    @opencl global_size=1000 float_add_kernel(a, one(T))
    @test OpenCL.@allowscalar(a[]) == T(1000)

    b = OpenCL.fill(T(1000))
    @opencl global_size=1000 float_sub_kernel(b, one(T))
    @test OpenCL.@allowscalar(b[]) == T(0)
end

end


# An atomic on global memory needs device scope to be atomic with respect to other
# work-groups; one on local memory only needs work-group scope.
#
# These kernels are only compiled. They store the result of the operation, as LLVM 15 turns
# an exchange whose result is unused into a store, and access memory through pointers, as
# `@inbounds` is ignored with --check-bounds=yes and the error path of a bounds check
# contains device-scope atomics.
function atomic_scope_kernel(op, a::AbstractArray{T}, val::T) where T
    unsafe_store!(pointer(a), op(pointer(a), val), 2)
    return
end
function atomic_scope_kernel_local(op, a::AbstractArray{T}, val::T) where T
    s = CLLocalArray(T, (1,))
    unsafe_store!(pointer(a), op(pointer(s), val))
    return
end

# the Scope operand of every atomic instruction in a SPIR-V disassembly
function atomic_scopes(asm)
    constants = Dict(m[1] => parse(UInt64, m[2]) for m in
                     eachmatch(r"(%\S+) = OpConstant %\S+ (\d+)", asm))
    [constants[m[2]] for m in eachmatch(r"= (OpAtomic\w+) %\S+ %\S+ (%\S+)", asm)]
end

@testset "atomic scopes ($backend)" for backend in (:llvm, :khronos)
    ops = [
        (OpenCL.atomic_add!,                         UInt32,  "OpAtomicIAdd"),
        (OpenCL.atomic_sub!,                         UInt32,  "OpAtomicISub"),
        ((p, v) -> OpenCL.atomic_inc!(p),            UInt32,  "OpAtomicIAdd"),
        ((p, v) -> OpenCL.atomic_dec!(p),            UInt32,  "OpAtomicISub"),
        (OpenCL.atomic_min!,                         Int32,   "OpAtomicSMin"),
        (OpenCL.atomic_max!,                         UInt32,  "OpAtomicUMax"),
        (OpenCL.atomic_and!,                         UInt32,  "OpAtomicAnd"),
        (OpenCL.atomic_or!,                          UInt32,  "OpAtomicOr"),
        (OpenCL.atomic_xor!,                         UInt32,  "OpAtomicXor"),
        (OpenCL.atomic_xchg!,                        UInt32,  "OpAtomicExchange"),
        ((p, v) -> OpenCL.atomic_cmpxchg!(p, v, v),  UInt32,  "OpAtomicCompareExchange"),
        (OpenCL.atomic_add!,                         Float32, "OpAtomic"),
        (OpenCL.atomic_sub!,                         Float32, "OpAtomic"),
        (OpenCL.atomic_min!,                         Float32, "OpAtomicCompareExchange"),
        (OpenCL.atomic_max!,                         Float32, "OpAtomicCompareExchange"),
        (OpenCL.atomic_xchg!,                        Float32, "OpAtomicExchange"),
        ((p, v) -> OpenCL.atomic_cmpxchg!(p, v, v),  Float32, "OpAtomicCompareExchange"),
        (SPIRVIntrinsics.atomic_add_native!,         Float32, "OpAtomic"),
        (SPIRVIntrinsics.atomic_min_fallback!,       Float32, "OpAtomicCompareExchange"),
    ]
    @testset "$inst ($op, $T)" for (op, T, inst) in ops
        for (kernel, scope) in ((atomic_scope_kernel, 1),          # Scope.Device
                                (atomic_scope_kernel_local, 2))    # Scope.Workgroup
            asm = sprint() do io
                OpenCL.code_native(io, kernel,
                                   Tuple{typeof(op), CLDeviceArray{T, 1, AS.CrossWorkgroup}, T};
                                   kernel=true, backend, dump_module=true)
            end
            @test occursin(inst, asm)
            scopes = atomic_scopes(asm)
            @test !isempty(scopes) && all(==(scope), scopes)
        end
    end
end

# A back-end whose method tables lack GPUCompiler's shared overlays (like OpenCL.jl before
# it declared `method_tables`) must still get the scopes SPIRVIntrinsics asks for.
struct LegacyCompilerParams <: GPUCompiler.AbstractCompilerParams end
GPUCompiler.runtime_module(::GPUCompiler.CompilerJob{<:Any,LegacyCompilerParams}) = OpenCL
GPUCompiler.method_table_view(job::GPUCompiler.CompilerJob{<:Any,LegacyCompilerParams}) =
    GPUCompiler.stack_method_tables(job.world, SPIRVIntrinsics.method_table)
GPUCompiler.isintrinsic(job::GPUCompiler.CompilerJob{<:Any,LegacyCompilerParams}, fn::String) =
    invoke(GPUCompiler.isintrinsic,
           Tuple{GPUCompiler.CompilerJob{GPUCompiler.SPIRVCompilerTarget}, typeof(fn)},
           job, fn) || contains(fn, "__spirv_")

@testset "atomic scopes without shared method tables" begin
    for (as, scope) in ((AS.CrossWorkgroup, 1), (AS.Workgroup, 2))
        f(p, v) = (SPIRVIntrinsics.atomic_add!(p, v); return)
        source = GPUCompiler.methodinstance(typeof(f), Tuple{LLVMPtr{UInt32,as}, UInt32})
        config = GPUCompiler.CompilerConfig(GPUCompiler.SPIRVCompilerTarget(),
                                            LegacyCompilerParams(); kernel=false)
        asm = sprint(io -> GPUCompiler.code_native(io, GPUCompiler.CompilerJob(source, config)))
        @test occursin("OpAtomicIAdd", asm)
        @test atomic_scopes(asm) == [scope]
    end
end


# The value each operation returns and stores, on global and local memory, compared with
# the same operation on the host. Floating-point values include NaNs and signed zeros:
# `atomic_min!` and `atomic_max!` behave like `min` and `max`, and `atomic_cmpxchg!`
# compares bit patterns.
function atomic_op_kernel(op, a, out, args...)
    @inbounds out[1] = op(pointer(a), args...)
    return
end
function atomic_op_kernel_local(op, a, out, args...)
    s = CLLocalArray(eltype(a), (1,))
    @inbounds s[1] = a[1]
    @inbounds out[1] = op(pointer(s), args...)
    @inbounds a[1] = s[1]
    return
end

# The atomics the compiler selects SPIR-V instructions for; without native floating-point
# addition, it uses compare-and-swap loops.
const default_atomics = OpenCL.compiler_config(dev).target.atomics
const no_fp_atomics = GPUCompiler.SPIRVAtomics(; default_atomics.int64)

function run_atomic_op(kernel, op, old::T, args...; atomics=nothing) where T
    a = OpenCL.fill(old)
    out = OpenCL.fill(T(42))
    if atomics === nothing
        @opencl global_size=1 kernel(op, a, out, args...)
    else
        @opencl global_size=1 atomics=atomics kernel(op, a, out, args...)
    end
    OpenCL.@allowscalar (a[], out[])
end

function test_atomic_op(op, model, cases; atomics=nothing)
    for kernel in (atomic_op_kernel, atomic_op_kernel_local), (old, args...) in cases
        stored, returned = run_atomic_op(kernel, op, old, args...; atomics)
        expected = model(old, args...)
        @test isequal(stored, expected)
        @test isequal(returned, old)
        isequal(stored, expected) && isequal(returned, old) ||
            @error "atomic operation returned or stored the wrong value" op kernel old args stored returned expected atomics
    end
end

int_atomic_ops = [
    # op, model, arity
    (OpenCL.atomic_add!,     +,                                   1),
    (OpenCL.atomic_sub!,     -,                                   1),
    (OpenCL.atomic_inc!,     old -> old + one(old),               0),
    (OpenCL.atomic_dec!,     old -> old - one(old),               0),
    (OpenCL.atomic_min!,     min,                                 1),
    (OpenCL.atomic_max!,     max,                                 1),
    (OpenCL.atomic_and!,     &,                                   1),
    (OpenCL.atomic_or!,      |,                                   1),
    (OpenCL.atomic_xor!,     xor,                                 1),
    (OpenCL.atomic_xchg!,    (old, val) -> val,                   1),
    (OpenCL.atomic_cmpxchg!, (old, cmp, val) -> old === cmp ? val : old, 2),
]
float_atomic_ops = [
    (OpenCL.atomic_add!,     +,                                   1),
    (OpenCL.atomic_sub!,     -,                                   1),
    (OpenCL.atomic_min!,     min,                                 1),
    (OpenCL.atomic_max!,     max,                                 1),
    (OpenCL.atomic_xchg!,    (old, val) -> val,                   1),
    (OpenCL.atomic_cmpxchg!, (old, cmp, val) -> old === cmp ? val : old, 2),
]

function int_cases(T, arity)
    arity == 0 && return [(T(5),), (typemax(T),), (typemin(T),)]
    arity == 2 && return [(T(5), T(5), T(7)), (T(5), T(4), T(7))]
    cases = [(T(5), T(3)), (T(3), T(5)), (typemax(T), one(T)), (T(0b1100), T(0b1010))]
    if T <: Signed
        push!(cases, (T(-2), T(7)), (T(7), T(-2)))
    end
    cases
end
function float_cases(T, arity)
    nan = T(NaN)
    if arity == 2
        return [(T(1), T(1), T(2)), (T(1), T(3), T(2)), (nan, nan, T(2)),
                (-zero(T), zero(T), T(2)), (zero(T), -zero(T), T(2))]
    end
    [(T(1.5), T(2.25)), (T(2.25), T(1.5)), (nan, T(1)), (T(1), nan),
     (-zero(T), zero(T)), (zero(T), -zero(T)), (-T(Inf), nan), (T(Inf), -T(Inf))]
end

supports_atomics(T) = sizeof(T) == 4 || "cl_khr_int64_base_atomics" in dev.extensions
supports_atomic_bitops(T) = sizeof(T) == 4 || "cl_khr_int64_extended_atomics" in dev.extensions

@testset "atomic operations ($T)" for T in integer_types
    supports_atomics(T) || continue
    @testset "$op" for (op, model, arity) in int_atomic_ops
        op in (OpenCL.atomic_min!, OpenCL.atomic_max!, OpenCL.atomic_and!,
               OpenCL.atomic_or!, OpenCL.atomic_xor!) && !supports_atomic_bitops(T) && continue
        test_atomic_op(op, model, int_cases(T, arity))
    end
end

# native addition on global memory only, and on local memory only
function fadd_atomics(T; global_add::Bool, local_add::Bool)
    prec = Dict(Float16 => "f16", Float32 => "f32", Float64 => "f64")[T]
    GPUCompiler.SPIRVAtomics(; default_atomics.int64,
                             Symbol("fadd_$(prec)_global") => global_add,
                             Symbol("fadd_$(prec)_local") => local_add)
end

@testset "atomic operations ($T, $name)" for T in float_types,
        (name, atomics) in (("default", nothing),
                            ("no native addition", no_fp_atomics),
                            ("native global addition", fadd_atomics(T; global_add=true, local_add=false)),
                            ("native local addition", fadd_atomics(T; global_add=false, local_add=true)))
    T == Float64 && !("cl_khr_fp64" in dev.extensions) && continue
    supports_atomics(T) || continue
    @testset "$op" for (op, model, arity) in float_atomic_ops
        # only addition depends on the descriptor
        atomics === nothing || op in (OpenCL.atomic_add!, OpenCL.atomic_sub!) || continue
        test_atomic_op(op, model, float_cases(T, arity); atomics)
    end

    # the IEEE minNum/maxNum instructions differ from `min`/`max`, so they're never used
    @testset "no native min/max" begin
        for op in (OpenCL.atomic_min!, OpenCL.atomic_max!)
            tt = Tuple{typeof(op), CLDeviceArray{T,0,AS.CrossWorkgroup},
                       CLDeviceArray{T,0,AS.CrossWorkgroup}, T}
            asm = sprint() do io
                OpenCL.code_native(io, atomic_op_kernel, tt; kernel=true, dump_module=true,
                                   atomics=something(atomics, default_atomics))
            end
            @test !occursin(r"OpAtomicF(Min|Max)EXT", asm)
        end
    end
end

# The compiler selects OpAtomicFAddEXT for floating-point addition on the address spaces the
# descriptor enables, and uses a compare-and-swap loop on the others.
fadd!(p, val) = SPIRVIntrinsics.atomic_modify!(p, +, val)
@testset "native addition per address space ($T)" for T in [Float16, Float32, Float64]
    T == Float16 && !("cl_khr_fp16" in dev.extensions) && continue
    T == Float64 && !("cl_khr_fp64" in dev.extensions) && continue
    T == Float64 && !default_atomics.int64 && continue
    for global_add in (false, true), local_add in (false, true)
        atomics = fadd_atomics(T; global_add, local_add)
        for (kernel, native) in ((atomic_scope_kernel, global_add),
                                 (atomic_scope_kernel_local, local_add))
            asm = sprint() do io
                OpenCL.code_native(io, kernel,
                                   Tuple{typeof(fadd!), CLDeviceArray{T, 1, AS.CrossWorkgroup}, T};
                                   kernel=true, dump_module=true, atomics)
            end
            @test occursin("OpAtomicFAddEXT", asm) == native
            @test occursin("OpAtomicCompareExchange", asm) == !native
            if T == Float16
                @test occursin("SPV_EXT_shader_atomic_float16_add", asm) == native
            end
        end
    end
end

@testset "atomics descriptor" begin
    source = OpenCL.program_backend() === :opencl || !("cl_khr_il_program" in dev.extensions)
    @test default_atomics === OpenCL.device_atomics(dev; source)
    if source
        # spirv2clc can't translate half-precision atomic addition
        @test !default_atomics.fadd_f16_global && !default_atomics.fadd_f16_local
        if OpenCL.max_opencl_c_version(dev) < v"2.0"
            @test default_atomics === no_fp_atomics
        end
    end

    # nothing enables native floating-point min/max
    @test !occursin("min_max", GPUCompiler.spirv_extensions(OpenCL.compiler_config(dev).target))

    # the user's extensions are kept, and the atomics add the ones they need
    config = OpenCL.compiler_config(dev; extensions=["SPV_KHR_expect_assume"],
                                    atomics=GPUCompiler.SPIRVAtomics(; fadd_f32_local=true))
    @test GPUCompiler.spirv_extensions(config.target) ==
          "+SPV_KHR_expect_assume,+SPV_EXT_shader_atomic_float_add"
end

# `@atomic` returns the old value when it uses an atomic operation, and the new one when it
# falls back to a compare-and-swap loop.
function atomic_arrayset_kernel(a, out, val)
    @inbounds out[1] = OpenCL.@atomic a[1] += val
    @inbounds out[2] = OpenCL.@atomic a[2] *= val
    return
end
@testset "@atomic ($T)" for T in [Int32, Float32]
    cases = T <: Integer ? [(T(3), T(2)), (T(-3), T(2))] :
                           [(T(3), T(2)), (-zero(T), T(2)), (T(NaN), T(2)), (T(Inf), zero(T))]
    for (old, val) in cases
        a = CLArray([old, old])
        out = CLArray(zeros(T, 2))
        @opencl global_size=1 atomic_arrayset_kernel(a, out, val)
        @test isequal(Array(a), [old + val, old * val])
        @test isequal(Array(out), [old, old * val])
    end

    asm = sprint() do io
        OpenCL.code_native(io, atomic_arrayset_kernel,
                           Tuple{CLDeviceArray{T,1,AS.CrossWorkgroup},
                                 CLDeviceArray{T,1,AS.CrossWorkgroup}, T};
                           kernel=true, dump_module=true)
    end
    scopes = atomic_scopes(asm)
    @test !isempty(scopes) && all(==(1), scopes)
end


# 8- and 16-bit atomics operate on the containing aligned 32-bit word, which has to be
# accessible: allocations are rounded up to a multiple of 4 bytes, and neighbouring elements
# in a word are updated concurrently without affecting each other.
const partword_types = filter([Int8, Int16, Float16]) do T
    T != Float16 || "cl_khr_fp16" in dev.extensions
end

@testset "partword allocations ($M)" for M in
        [Dict(cl.USMBackend() => cl.UnifiedDeviceMemory,
              cl.SVMBackend() => cl.SharedVirtualMemory,
              cl.BufferBackend() => cl.Buffer)[b] for b in cl.supported_memory_backends(dev)]
    for T in partword_types, n in 1:5
        a = CLVector{T, M}(undef, n)
        @test length(a) == n
        @test sizeof(a) == n * sizeof(T)
        @test sizeof(a.data[]) % 4 == 0
        @test sizeof(a.data[]) >= n * sizeof(T)
        @test UInt(pointer(a)) % 4 == 0

        # resizing allocates anew
        resize!(a, n + 1)
        @test sizeof(a) == (n + 1) * sizeof(T)
        @test sizeof(a.data[]) % 4 == 0
    end
    @test sizeof(CLVector{Int8, M}(undef, 0).data[]) == 0
end

function partword_contention_kernel(a, iters)
    p = pointer(a, (get_global_id() - 1) % length(a) + 1)
    for _ in 1:iters
        SPIRVIntrinsics.atomic_modify!(p, +, one(eltype(a)))
    end
    return
end
function partword_contention_kernel_local(a, iters, ::Val{n}) where {n}
    s = CLLocalArray(eltype(a), (n,))
    i = get_local_id()
    i <= n && @inbounds(s[i] = zero(eltype(a)))
    barrier(OpenCL.LOCAL_MEM_FENCE)
    p = pointer(s, (i - 1) % n + 1)
    for _ in 1:iters
        SPIRVIntrinsics.atomic_modify!(p, +, one(eltype(a)))
    end
    barrier(OpenCL.LOCAL_MEM_FENCE)
    i <= n && @inbounds(a[i] = s[i])
    return
end

@testset "partword contention ($T)" for T in partword_types
    # every element is incremented by 8 work-items, `iters` times; 64 is exact for all types
    iters = 8
    for n in 1:5
        a = OpenCL.zeros(T, n)
        @opencl global_size=8n partword_contention_kernel(a, iters)
        @test Array(a) == fill(T(64), n)

        b = OpenCL.zeros(T, n)
        @opencl global_size=8n local_size=8n partword_contention_kernel_local(b, iters, Val(n))
        @test Array(b) == fill(T(64), n)
    end
end
