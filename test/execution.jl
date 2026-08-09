using SPIRV_LLVM_Translator_jll
using IOCapture
using KernelAbstractions

@testset "@opencl" begin

dummy() = nothing

@test_throws UndefVarError @opencl undefined()
@test_throws MethodError @opencl dummy(1)


@testset "launch configuration" begin
    @opencl dummy()

    global_size = 1
    @opencl global_size dummy()
    @opencl global_size=1 dummy()
    @opencl global_size=(1,1) dummy()
    @opencl global_size=(1,1,1) dummy()

    local_size = 1
    @opencl global_size local_size dummy()
    @opencl global_size=1 local_size=1 dummy()
    @opencl global_size=(1,1) local_size=(1,1) dummy()
    @opencl global_size=(1,1,1) local_size=(1,1,1) dummy()

    @test_throws ArgumentError @opencl global_size=(1,) local_size=(1,1) dummy()
    @test_throws InexactError @opencl global_size=(-2) dummy()
    @test_throws InexactError @opencl local_size=(-2) dummy()
end

@testset "launch=false" begin
    # XXX: how are svm_pointers handled here?
    k = @opencl launch=false dummy()
    k()
    k(; global_size=1)
end

@testset "inference" begin
    foo() = @opencl dummy()
    @inferred foo()

    # with arguments, we call OpenCL.kernel_convert
    kernel(a) = return
    bar(a) = @opencl kernel(a)
    @inferred bar(CLArray([1]))
end


@testset "reflection" begin
    OpenCL.code_lowered(dummy, Tuple{})
    OpenCL.code_typed(dummy, Tuple{})
    OpenCL.code_warntype(devnull, dummy, Tuple{})
    OpenCL.code_llvm(devnull, dummy, Tuple{})
    OpenCL.code_native(devnull, dummy, Tuple{})

    @device_code_lowered @opencl dummy()
    @device_code_typed @opencl dummy()
    @device_code_warntype io=devnull @opencl dummy()
    @device_code_llvm io=devnull @opencl dummy()
    @device_code_native io=devnull @opencl dummy()

    mktempdir() do dir
        @device_code dir=dir @opencl dummy()
    end

    @test_throws ErrorException @device_code_lowered nothing

    # make sure kernel name aliases are preserved in the generated code
    @test occursin("dummy", sprint(io->(@device_code_llvm io=io optimize=false @opencl dummy())))
    @test occursin("dummy", sprint(io->(@device_code_llvm io=io @opencl dummy())))
    @test occursin("dummy", sprint(io->(@device_code_native io=io @opencl dummy())))

    # make sure invalid kernels can be partially reflected upon
    let
        invalid_kernel() = throw()
        @test_throws OpenCL.InvalidIRError @opencl invalid_kernel()
        @test_throws OpenCL.InvalidIRError IOCapture.capture() do
            @device_code_warntype @opencl invalid_kernel()
        end
        c = IOCapture.capture() do
            try
                @device_code_warntype @opencl invalid_kernel()
            catch
            end
        end
        @test occursin("Body::Union{}", c.output)
    end

    # set name of kernel
    @test occursin("mykernel", sprint(io->(@device_code_llvm io=io begin
        @opencl name="mykernel" dummy()
    end)))

    @test OpenCL.return_type(identity, Tuple{Int}) === Int
    @test OpenCL.return_type(sin, Tuple{Float32}) === Float32
    @test OpenCL.return_type(getindex, Tuple{CLDeviceArray{Float32,1,AS.CrossWorkgroup},Int32}) === Float32
    @test OpenCL.return_type(getindex, Tuple{Base.RefValue{Integer}}) === Integer
end

end

###############################################################################

@testset "argument passing" begin

function memset(a, val)
    gid = get_global_id(1)
    @inbounds a[gid] = val
    return
end

a = CLArray{Int}(undef, 10)
@opencl global_size=length(a) memset(a, 42)
@test all(Array(a) .== 42)

end

@kernel cpu=false function partial_workgroup_localmem!(out, pred, @Const(v))
    temp = @localmem Int8 (1,)
    i = @index(Global, Linear)

    temp[1] = 0
    @synchronize()

    if pred(v[i])
        temp[1] = 1
    end

    @synchronize()
    if temp[1] != 0
        out[1] = 1
    end
end

@testset "partial workgroup local memory" begin
    backend = OpenCLBackend()
    v = CLArray(zeros(Float32, 16))

    # The old PoCL lowering was nondeterministic across launches.
    for _ in 1:10
        out = KernelAbstractions.zeros(backend, Int8, 1)
        partial_workgroup_localmem!(backend, 256)(out, x -> x < 0, v; ndrange=length(v))
        KernelAbstractions.synchronize(backend)
        @test only(Array(out)) == 0
    end
end

@kernel cpu=false unsafe_indices=true function scatter_partial_workgroup!(
    dest, src, offsets,
)
    @uniform N = @groupsize()[1]
    values = @localmem eltype(src) (N,)
    digits = @localmem UInt32 (N,)
    bases = @localmem UInt32 (256,)

    group = Int(@index(Group, Linear)) - 1
    lane = Int(@index(Local, Linear)) - 1
    len = Int(length(src))
    groups = Int(length(offsets)) ÷ 256
    i = group * Int(N) + lane

    if i < len
        values[lane + 1] = src[i + 1]
    end
    j = lane
    while j < 256
        bases[j + 1] = offsets[j * groups + group + 1]
        j += Int(N)
    end
    @synchronize()

    digit = UInt32(i < len ? values[lane + 1] & 0xff : 0)
    digits[lane + 1] = digit
    @synchronize()

    if i < len
        rank = UInt32(0)
        for j in UInt32(1):UInt32(lane)
            rank += UInt32(digits[j] == digit)
        end
        dest[Int(bases[digit + 1]) + Int(rank) + 1] = values[lane + 1]
    end
end

@testset "partial workgroup scatter" begin
    backend = OpenCLBackend()
    input = UInt32[1, 0, 0]
    src = CLArray(input)

    counts = zeros(UInt32, 256 * 2)
    for group in 0:1
        for i in 2group:min(length(input), 2group + 2) - 1
            digit = input[i + 1] & 0xff
            counts[Int(digit) * 2 + group + 1] += 1
        end
    end
    offsets = CLArray(vcat(UInt32(0), cumsum(counts)[1:end-1]))

    for _ in 1:10
        dest = similar(src)
        scatter_partial_workgroup!(backend, 2)(dest, src, offsets; ndrange=4)
        KernelAbstractions.synchronize(backend)
        @test Array(dest) == sort(input)
    end
end

const SCAN_LOG_NUM_BANKS = 5
@inline scan_conflict_free_offset(n) = n >> SCAN_LOG_NUM_BANKS

@kernel cpu=false inbounds=true unsafe_indices=true function scan_blocks!(
    op, values, init, neutral, inclusive, flags, prefixes,
)
    len = length(values)
    @uniform block_size = @groupsize()[1]
    temp = @localmem eltype(values) (
        2block_size + scan_conflict_free_offset(2block_size),
    )

    group = @index(Group, Linear) - 1
    lane = @index(Local, Linear) - 1
    groups = @ndrange()[1] ÷ block_size
    block_offset = group * block_size * 2
    ai = lane
    bi = lane + block_size
    bank_offset_a = scan_conflict_free_offset(ai)
    bank_offset_b = scan_conflict_free_offset(bi)

    temp[ai + bank_offset_a + 1] =
        block_offset + ai < len ? values[block_offset + ai + 1] : neutral
    temp[bi + bank_offset_b + 1] =
        block_offset + bi < len ? values[block_offset + bi + 1] : neutral

    offset = typeof(lane)(1)
    next_pow2 = 2block_size
    d = next_pow2 >> 1
    while d > 0
        @synchronize()
        if lane < d
            a = offset * (2lane + 1) - 1
            b = offset * (2lane + 2) - 1
            a += scan_conflict_free_offset(a)
            b += scan_conflict_free_offset(b)
            temp[b + 1] = op(temp[b + 1], temp[a + 1])
        end
        offset <<= 1
        d >>= 1
    end

    if lane == 0
        root = next_pow2 - 1
        temp[root + scan_conflict_free_offset(root) + 1] = group == 0 ? init : neutral
    end

    d = typeof(lane)(1)
    while d < next_pow2
        offset >>= 1
        @synchronize()
        if lane < d
            a = offset * (2lane + 1) - 1
            b = offset * (2lane + 2) - 1
            a += scan_conflict_free_offset(a)
            b += scan_conflict_free_offset(b)
            t = temp[a + 1]
            temp[a + 1] = temp[b + 1]
            temp[b + 1] = op(temp[b + 1], t)
        end
        d <<= 1
    end

    if inclusive || (group != 0 && !isnothing(flags))
        @synchronize()
        first_value = temp[ai + bank_offset_a + 1]
        second_value = temp[bi + bank_offset_b + 1]
        @synchronize()

        if ai > 0
            temp[ai - 1 + scan_conflict_free_offset(ai - 1) + 1] = first_value
        end
        temp[bi - 1 + scan_conflict_free_offset(bi - 1) + 1] = second_value
        if bi == 2block_size - 1
            if group < groups - 1
                temp[bi + bank_offset_b + 1] =
                    op(second_value, values[(group + 1) * block_size * 2])
            else
                temp[bi + bank_offset_b + 1] = op(second_value, values[len])
            end
        end
    end

    @synchronize()

    if bi == 2block_size - 1 && !isnothing(prefixes)
        if isnothing(flags) && !inclusive
            last_global = block_offset + bi
            prefixes[group + 1] = last_global < len ?
                op(temp[bi + bank_offset_b + 1], values[last_global + 1]) :
                temp[bi + bank_offset_b + 1]
        else
            prefixes[group + 1] = temp[bi + bank_offset_b + 1]
        end
    end

    if block_offset + ai < len
        values[block_offset + ai + 1] = temp[ai + bank_offset_a + 1]
    end
    if block_offset + bi < len
        values[block_offset + bi + 1] = temp[bi + bank_offset_b + 1]
    end
end

@testset "local memory loads across barriers" begin
    backend = OpenCLBackend()
    input = UInt32.(1:128)
    values = CLArray(input)
    prefixes = similar(values, 1)

    # Both launches are needed to exercise context handling across barriers.
    kernel = scan_blocks!(backend, 64)
    kernel(+, values, UInt32(0), UInt32(0), false, nothing, prefixes; ndrange=64)
    KernelAbstractions.synchronize(backend)
    kernel(+, prefixes, UInt32(0), UInt32(0), true, nothing, nothing; ndrange=64)
    KernelAbstractions.synchronize(backend)

    @test Array(values) == vcat(UInt32(0), cumsum(input)[1:end-1])
    @test only(Array(prefixes)) == sum(input)
end

@testset "broadcasting" begin
    a = rand(Float32, 2, 3)
    b = rand(Float32, 2)

    c = a .+ b
    a_cl, b_cl = CLArray(a), CLArray(b)
    c_cl = a_cl .+ b_cl
    @test Array(c_cl) == c
    @test c_cl isa CLArray{Float32, 2, OpenCL.memory_type()}

    if cl.usm_supported(cl.device())
        a_cl, b_cl = CLMatrix{Float32, cl.UnifiedSharedMemory}(a), CLVector{Float32, OpenCL.memory_type()}(b)
        c_cl = a_cl .+ b_cl
        @test Array(c_cl) == c
        @test c_cl isa CLArray{Float32, 2, cl.UnifiedSharedMemory}
    end
end

@testset "backends" begin
    llvm_backend_llvm = sprint() do io
        OpenCL.code_llvm(io, () -> nothing, (); dump_module = true, backend = :llvm)
    end
    if Int === Int64
        @test occursin("target triple = \"spirv64v1.4-unknown-unknown-unknown\"", llvm_backend_llvm)
    end

    llvm_backend_khronos = sprint() do io
        OpenCL.code_llvm(io, () -> nothing, (); dump_module = true, backend = :khronos)
    end
    if Int === Int64
        @test occursin("target triple = \"spir64-unknown-unknown\"", llvm_backend_khronos)
    end
end

@testset "has_feature folding" begin
    # has_feature must fold at compile time: the feature-bitset global gets baked in and DCE'd,
    # leaving only the branch matching the device.
    function feature_select(a)
        @inbounds a[] = has_feature(:subgroups) ? Int32(12345) : Int32(54321)
        return
    end
    ir = sprint(io -> OpenCL.code_llvm(io, feature_select,
                                       Tuple{CLDeviceArray{Int32, 0, AS.CrossWorkgroup}};
                                       kernel = true))
    @test !occursin("__opencl_feature_bitset", ir)
    if OpenCL.feature_supported(cl.device(), :subgroups)
        @test occursin("12345", ir) && !occursin("54321", ir)
    else
        @test occursin("54321", ir) && !occursin("12345", ir)
    end
end

@testset "compilation cache" begin
    mod = @eval module $(gensym())
        @noinline child() = return
        kernel() = child()
    end

    count() = OpenCL.compilations[]
    launch() = @opencl mod.kernel()

    # the initial launch compiles
    n = count()
    Base.invokelatest(launch)
    @test count() == n+1

    # a second launch hits the cache
    Base.invokelatest(launch)
    @test count() == n+1

    # jobs differing only in codegen-level settings get their own artifacts...
    OpenCL.clfunction(mod.kernel, Tuple{}; name="custom")
    @test count() == n+2
    # ... which are cached as well
    OpenCL.clfunction(mod.kernel, Tuple{}; name="custom")
    @test count() == n+2

    # reflection observes already-compiled kernels (by forcing recompilation,
    # which must leave the cached entry in a usable state)
    @test !isempty(sprint(io->(@device_code_llvm io=io Base.invokelatest(launch))))
    n = count()
    Base.invokelatest(launch)
    @test count() == n

    # redefining the kernel recompiles...
    @eval mod kernel() = (child(); child())
    Base.invokelatest(launch)
    @test count() == n+1
    # ... as does redefining a callee
    @eval mod @noinline child() = nothing
    Base.invokelatest(launch)
    @test count() == n+2
end
