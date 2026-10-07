using LinearAlgebra
import Adapt

@testset "constructors" begin
    xs = CLArray{Int, 2, cl.Buffer}(undef, 2, 3)
    @test collect(CLArray([1 2; 3 4])) == [1 2; 3 4]
    @test testf(vec, rand(Float32, 5, 3))
    @test Base.elsize(xs) == sizeof(Int)
    @test CLArray{Int, 2}(xs) === xs

    @test device_accessible(xs)
    @test !host_accessible(xs)
    @test_throws ArgumentError Base.unsafe_convert(Ptr{Int}, xs)
    @test_throws ArgumentError Base.unsafe_convert(Ptr{Float32}, xs)

    @test collect(OpenCL.zeros(Float32, 2, 2)) == zeros(Float32, 2, 2)
    @test collect(OpenCL.ones(Float32, 2, 2)) == ones(Float32, 2, 2)

    @test collect(OpenCL.fill(0, 2, 2)) == zeros(Int, 2, 2)
    @test collect(OpenCL.fill(1, 2, 2)) == ones(Int, 2, 2)
end

@testset "adapt" begin
    A = rand(Float32, 3, 3)
    dA = CLArray(A)
    @test Adapt.adapt(Array, dA) == A
    @test Adapt.adapt(CLArray, A) isa CLArray
    @test Array(Adapt.adapt(CLArray, A)) == A
end

@testset "reshape" begin
    A = [
        1 2 3 4
        5 6 7 8
    ]
    gA = reshape(CLArray(A), 1, 8)
    _A = reshape(A, 1, 8)
    _gA = Array(gA)
    @test all(_A .== _gA)
    A = [1, 2, 3, 4]
    gA = reshape(CLArray(A), 4)
end

@testset "fill(::SubArray)" begin
    xs = OpenCL.zeros(Float32, 3)
    fill!(view(xs, 2:2), 1)
    @test Array(xs) == [0, 1, 0]
end

@testset "fill! with sizes unsupported by the OpenCL fill commands" begin
    # patterns must be 1, 2, 4, ..., 128 bytes
    for (T, val) in ((NTuple{3, UInt8}, (0x01, 0x02, 0x03)),
                     (NTuple{5, Int32}, ntuple(Int32, 5)),
                     (NTuple{48, Int32}, ntuple(Int32, 48)),
                     (Nothing, nothing))
        xs = CLArray{T}(undef, 5)
        fill!(xs, val)
        @test Array(xs) == fill(val, 5)
    end
end

@testset "reinterpret of view with non-aligned offset" begin
    # reinterpreting a view to a larger element type where the byte offset
    # is not a multiple of the new element size
    a = CLArray(Int32[1,2,3,4,5,6,7,8,9])
    v = view(a, 2:7)  # offset of 1 Int32 = 4 bytes
    r = reinterpret(Int64, v)  # Int64 = 8 bytes; 4 is not a multiple of 8
    @test Array(r) == reinterpret(Int64, @view Array(a)[2:7])
end
# TODO: Look into how to port the @sync

if cl.USMBackend() in cl.supported_memory_backends(cl.device())
    @testset "shared buffers & unsafe_wrap" begin
        a = CLVector{Int, cl.UnifiedSharedMemory}(undef, 2)

        # check that basic operations work on arrays backed by shared memory
        fill!(a, 40)
        a .+= 2
        @test Array(a) == [42, 42]

        # derive an Array object and test that the memory keeps in sync
        b = unsafe_wrap(Array, a)
        b[1] = 100
        @test Array(a) == [100, 42]
        copyto!(a, 2, [200], 1, 1)
        cl.finish(cl.queue())
        @test b == [100, 200]
    end

    # https://github.com/JuliaGPU/CUDA.jl/issues/2191
    @testset "preserving memory types" begin
        a = CLVector{Int, cl.UnifiedSharedMemory}([1])
        @test OpenCL.memtype(a) == cl.UnifiedSharedMemory

        # unified-ness should be preserved
        b = a .+ 1
        @test OpenCL.memtype(b) == cl.UnifiedSharedMemory

        # when there's a conflict, we should defer to unified memory
        c = CLVector{Int, cl.UnifiedSharedMemory}([1])
        d = CLVector{Int, cl.UnifiedDeviceMemory}([1])
        e = c .+ d
        @test OpenCL.memtype(e) == cl.UnifiedSharedMemory
    end
end

@testset "merging memory types in broadcasts" begin
    style(M) = OpenCL.CLArrayStyle{1, M}()
    merged(M1, M2) = Base.Broadcast.BroadcastStyle(style(M1), style(M2))
    @test merged(cl.Buffer, cl.Buffer) == style(cl.Buffer)
    @test merged(cl.UnifiedDeviceMemory, cl.UnifiedHostMemory) == style(cl.UnifiedSharedMemory)
    @test merged(cl.UnifiedDeviceMemory, cl.SharedVirtualMemory) == style(cl.UnifiedSharedMemory)
    # without USM, fall back to SVM, which is also accessible from both host and device
    @test merged(cl.Buffer, cl.SharedVirtualMemory) == style(cl.SharedVirtualMemory)
end

function wrap_kernel(a)
    i = get_global_id()
    @inbounds a[i] = i
    return
end

@testset "wrapping host memory" begin
    M = OpenCL.system_memory_type()
    if M === nothing
        @test_throws ArgumentError unsafe_wrap(CLArray, Float32[1])
    else
        a = Float32[1, 2, 3, 4]
        b = unsafe_wrap(CLArray, a)
        @test b isa CLVector{Float32, M}
        @test size(b) == size(a)
        @test UInt(pointer(b)) == UInt(pointer(a))
        @test host_accessible(b) && device_accessible(b)

        # changes are visible in both directions
        b .+= 1
        cl.finish(cl.queue())
        @test a == [2, 3, 4, 5]
        a[1] = 10
        @test Array(b) == [10, 3, 4, 5]
        @test b[1] == 10

        # wrapping the wrapper again gives back the original memory
        @test pointer(unsafe_wrap(Array, b)) == pointer(a)

        for AT in [CLArray, CLArray{Float32}, CLArray{Float32, 1}, CLArray{Float32, 1, M}],
            f in [x -> unsafe_wrap(AT, pointer(x), length(x)),
                  x -> unsafe_wrap(AT, pointer(x), size(x)),
                  x -> unsafe_wrap(AT, x)]
            c = f(a)
            @test c isa CLVector{Float32, M}
            @test Array(c) == a
        end
        let m = rand(Float32, 3, 4)
            c = unsafe_wrap(CLArray, m)
            @test c isa CLMatrix{Float32, M}
            @test Array(c) == m
        end
        @test isempty(Array(unsafe_wrap(CLArray, Float32[])))

        # allocating broadcasts, also when mixing with regular arrays
        @test Array(b .* 2) == 2 .* a
        @test Array(b .+ CLArray(a)) == 2 .* a

        # copies between wrapped memory, host arrays, and regular device arrays
        c = CLArray{Float32}(undef, 4)
        copyto!(c, b)
        @test Array(c) == a
        copyto!(b, CLArray(Float32[5, 6, 7, 8]))
        cl.finish(cl.queue())
        @test a == [5, 6, 7, 8]
        copyto!(b, Float32[1, 2, 3, 4])
        @test a == [1, 2, 3, 4]
        copyto!(b, unsafe_wrap(CLArray, Float32[4, 3, 2, 1]))
        @test a == [4, 3, 2, 1]

        fill!(b, 42)
        cl.finish(cl.queue())
        @test all(==(42), a)
        view(b, 2:3) .= 0
        cl.finish(cl.queue())
        @test a == [42, 0, 0, 42]
        @test sum(b) == 84

        @opencl global_size=length(b) wrap_kernel(b)
        cl.finish(cl.queue())
        @test a == [1, 2, 3, 4]

        prog = cl.Program(source="""
            __kernel void add_one(__global float *a) {
                a[get_global_id(0)] += 1;
            }""") |> cl.build!
        clcall(cl.Kernel(prog, "add_one"), Tuple{CLPtr{Float32}}, b; global_size=length(b))
        cl.finish(cl.queue())
        @test a == [2, 3, 4, 5]

        # elements that are only aligned to part of their size
        bytes = zeros(UInt8, 20)
        GC.@preserve bytes begin
            c = unsafe_wrap(CLArray, Ptr{NTuple{2, Float32}}(pointer(bytes) + 4), 2)
            fill!(c, (1.0f0, 2.0f0))
            cl.finish(cl.queue())
        end
        @test reinterpret(Float32, bytes[5:20]) == [1, 2, 1, 2]

        # the wrapper keeps the array alive
        c = unsafe_wrap(CLArray, fill(1.0f0, 1024))
        GC.gc(true)
        @test sum(c) == 1024

        @test_throws ArgumentError resize!(b, 5)
        @test_throws ArgumentError unsafe_wrap(CLVector{Float32, cl.UnifiedDeviceMemory}, a)
        @test_throws ArgumentError unsafe_wrap(CLArray, Ptr{Float32}(C_NULL), 1)
        @test_throws ArgumentError unsafe_wrap(CLArray, pointer(a), (-1,))
        GC.@preserve bytes begin
            @test_throws ArgumentError unsafe_wrap(CLArray, Ptr{Float32}(pointer(bytes) + 1), 1)
        end
    end
end

@testset "resizing" begin
    a = CLArray([1, 2, 3])

    resize!(a, 3)
    @test length(a) == 3
    @test Array(a) == [1, 2, 3]

    resize!(a, 5)
    @test length(a) == 5
    @test Array(a)[1:3] == [1, 2, 3]

    resize!(a, 2)
    @test length(a) == 2
    @test Array(a)[1:2] == [1, 2]

    b = CLArray{Int}(undef, 0)
    @test length(b) == 0
    resize!(b, 1)
    @test length(b) == 1
end

# finalizers run in no particular order, e.g. at exit (JuliaGPU/OpenCL.jl#279), so memory
# has to remain freeable after the queue and context it was allocated with are finalized
@testset "freeing after finalizing its queue and context" begin
    memtypes = Dict(cl.USMBackend() => cl.UnifiedDeviceMemory,
                    cl.SVMBackend() => cl.SharedVirtualMemory,
                    cl.BufferBackend() => cl.Buffer)
    @testset "$M" for M in [memtypes[b] for b in cl.supported_memory_backends(cl.device())]
        ctx = cl.Context(cl.device())
        cl.context!(ctx) do
            queue = cl.CmdQueue()
            cl.queue!(queue) do
                a = CLVector{Float32, M}(ones(Float32, 1024))
                # leave coarse-grained SVM mapped, so that freeing it needs the queue
                M == cl.SharedVirtualMemory && unsafe_wrap(Array, a)

                # get rid of temporary objects that also reference the context
                GC.gc()

                finalize(queue)
                finalize(ctx)
                @test (OpenCL.unsafe_free!(a); true)
            end
        end
    end
end

# finalizers can run on a task that is bound to a different platform
let other = findfirst(p -> p != cl.platform() && !isempty(cl.devices(p)), cl.platforms())
    if other !== nothing && cl.USMBackend() in cl.supported_memory_backends(cl.device())
        @testset "freeing USM from a task bound to another platform" begin
            a = CLVector{Float32, cl.UnifiedDeviceMemory}(ones(Float32, 1024))
            cl.device!(first(cl.devices(cl.platforms()[other]))) do
                @test (OpenCL.unsafe_free!(a); true)
            end
        end
    end
end
