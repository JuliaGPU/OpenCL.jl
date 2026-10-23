# arrays of isbits unions, which keep one selector byte per element past the data

# a struct with a union field is not isbits, but is stored inline with its own selector
struct UnionField
    x::Union{Int, Float32}
end

const union_arrays = [
    Union{Int, Nothing}[1, nothing, 3, nothing],
    # members of different size and alignment
    Union{Int8, Int64}[Int8(1), 2, Int8(3), 4],
    Union{NTuple{3, Int8}, Int16}[Int8.((1, 2, 3)), Int16(4), Int8.((5, 6, 7)), Int16(8)],
    # the element size (12 bytes) is not a multiple of the alignment (8 bytes)
    Union{Int64, NTuple{3, Int32}}[1, Int32.((2, 3, 4)), 5, Int32.((6, 7, 8))],
    # only singletons, so no data, only selector bytes
    Union{Nothing, Missing}[nothing, missing, missing, nothing],
    # nested unions are flattened
    Union{Union{Missing, Nothing}, Int}[0, missing, nothing, 3],
]

# the memory types arrays can be allocated with, and which of those kernels can use
# (host memory has no device address, and neither do buffers without the BDA extension)
const union_memtypes, union_kernel_memtypes = let backends = cl.supported_memory_backends(cl.device())
    usm = [cl.UnifiedDeviceMemory, cl.UnifiedSharedMemory, cl.UnifiedHostMemory]
    all = [cl.Buffer]
    cl.USMBackend() in backends && append!(all, usm)
    cl.SVMBackend() in backends && push!(all, cl.SharedVirtualMemory)
    all, filter(all) do M
        M != cl.UnifiedHostMemory && (M != cl.Buffer || cl.BufferBackend() in backends)
    end
end

function bump!(x)
    i = get_global_id()
    val = x[i]
    if val !== nothing
        x[i] = val + 1
    end
    return
end

@testset "size" begin
    a = CLArray(Union{Int, Nothing}[1, nothing, 3])
    @test sizeof(a) == 3 * sizeof(Int)
end

@testset "copies ($M)" for M in union_memtypes
    for a in union_arrays
        T = eltype(a)
        b = CLVector{T, M}(a)
        @test isequal(Array(b), a)
        @test isequal(Array(copy(b)), a)

        c = similar(b)
        copyto!(c, b)
        @test isequal(Array(c), a)
        copyto!(c, 2, b, 1, 3)
        @test isequal(Array(c), [a[1]; a[1:3]])

        # partial copies to and from the host
        copyto!(c, 3, a, 1, 2)
        @test isequal(Array(c), [a[1]; a[1]; a[1:2]])
        d = Vector{T}(undef, 4)
        copyto!(d, 2, b, 3, 2)
        @test isequal(d[2:3], a[3:4])
    end
end

@testset "views ($M)" for M in union_memtypes
    # the data is chosen so that the parent's first selector bytes differ from the view's
    a = Union{Int, Nothing}[nothing, nothing, 1, nothing, 3, 5]
    b = CLVector{eltype(a), M}(a)
    v = view(b, 3:5)
    @test Array(v) == view(a, 3:5)
    @test Array(copy(v)) == view(a, 3:5)

    copyto!(view(b, 1:2), [1, nothing])
    @test Array(b) == [1, nothing, 1, nothing, 3, 5]
    copyto!(view(b, 5:6), view(b, 1:2))
    @test Array(b) == [1, nothing, 1, nothing, 1, nothing]
    copyto!(b, [nothing, nothing, 1, nothing, 3, 5])

    # derived from views
    @test Array(view(view(b, 2:6), 2:4)) == [1, nothing, 3]
    @test Array(reshape(view(b, 3:6), 2, 2)) == [1 3; nothing 5]
    @test Array(vec(view(reshape(b, 2, 3), :, 2))) == [1, nothing]

    # unions of singletons have no byte offset, but views still have their own selectors
    c = CLVector{Union{Nothing, Missing}, M}([nothing, nothing, missing, missing])
    w = view(c, 3:4)
    @test isequal(Array(w), [missing, missing])
    @test isequal(Array(copy(w)), [missing, missing])
end

@testset "kernels ($M)" for M in union_kernel_memtypes
    a = CLVector{Union{Int, Nothing}, M}([1, nothing, 3])
    @opencl global_size=3 bump!(a)
    @test Array(a) == [2, nothing, 4]

    # a view must use its own selector bytes, and leave the rest of its parent alone
    b = CLVector{Union{Int, Nothing}, M}([nothing, nothing, 1, nothing, 3, 5])
    v = view(b, 3:5)
    @test Array(map(x -> something(x, 0), v)) == [1, 0, 3]
    @opencl global_size=3 bump!(v)
    @test Array(v) == [2, nothing, 4]
    @test Array(b) == [nothing, nothing, 2, nothing, 4, 5]

    # non-contiguous views index into their parent
    @opencl global_size=3 bump!(view(b, 1:2:5))
    @test Array(b) == [nothing, nothing, 3, nothing, 5, 5]

    function flip!(ok, x)
        i = get_global_id()
        ok[i] = x[i] === missing
        x[i] = x[i] === missing ? nothing : missing
        return
    end
    c = CLVector{Union{Nothing, Missing}, M}([nothing, nothing, missing, missing])
    ok = CLVector{Bool, M}(undef, 2)
    @opencl global_size=2 flip!(ok, view(c, 2:3))
    @test Array(ok) == [false, true]
    @test isequal(Array(c), [nothing, missing, nothing, missing])
    fill!(view(c, 2:3), missing)
    @test isequal(Array(c), [nothing, missing, missing, missing])

    function bump_nested!(x)
        i = get_global_id()
        val = x[i]
        if val isa Int
            x[i] = val + 1
        end
        return
    end
    d = CLVector{Union{Union{Missing, Nothing}, Int}, M}([0, missing, nothing])
    @opencl global_size=3 bump_nested!(d)
    @test isequal(Array(d), [1, missing, nothing])

    # members of different size and alignment
    function widen!(x)
        i = get_global_id()
        val = x[i]
        if val isa NTuple{3, Int32}
            x[i] = Int64(sum(val))
        else
            x[i] = (Int32(val), Int32(0), Int32(0))
        end
        return
    end
    e = CLVector{Union{Int64, NTuple{3, Int32}}, M}([1, Int32.((2, 3, 4)), 5])
    @opencl global_size=3 widen!(e)
    @test Array(e) == Union{Int64, NTuple{3, Int32}}[Int32.((1, 0, 0)), 9, Int32.((5, 0, 0))]

    # a struct with a union field keeps its selector inside the element
    function fill_struct!(x)
        i = get_global_id()
        x[i] = i == 1 ? UnionField(1) : UnionField(2.0f0)
        return
    end
    f = CLVector{UnionField, M}(undef, 2)
    @opencl global_size=2 fill_struct!(f)
    @test Array(f) == [UnionField(1), UnionField(2.0f0)]
    function copy_struct!(y, x)
        i = get_global_id()
        y[i] = x[i]
        y[i + 2] = Base.Experimental.Const(x)[i]
        return
    end
    g = CLVector{UnionField, M}(undef, 4)
    @opencl global_size=2 copy_struct!(g, f)
    @test Array(g) == [UnionField(1), UnionField(2.0f0), UnionField(1), UnionField(2.0f0)]

    # their selector isn't stored past the elements, nor loaded from there
    struct_array(bytes) =
        CLDeviceArray((2,), reinterpret(Core.LLVMPtr{UnionField, AS.CrossWorkgroup}, pointer(bytes)))
    function store_struct!(bytes)
        i = get_global_id()
        struct_array(bytes)[i] = UnionField(i)
        return
    end
    function load_struct!(y, bytes)
        i = get_global_id()
        y[i] = struct_array(bytes)[i]
        return
    end
    bytes = CLVector{UInt8, M}(fill(0xff, 3 * sizeof(UnionField)))
    @opencl global_size=2 store_struct!(bytes)
    @test all(==(0xff), Array(bytes)[2 * sizeof(UnionField) + 1:end])
    @opencl global_size=2 load_struct!(g, bytes)
    @test Array(g)[1:2] == [UnionField(1), UnionField(2)]
end

@testset "local memory" begin
    function reverse_local!(x)
        i = get_local_id()
        tmp = CLLocalArray(Union{Int64, NTuple{3, Int32}}, 4)
        tmp[i] = x[i]
        barrier(OpenCL.LOCAL_MEM_FENCE)
        x[i] = tmp[5 - i]
        return
    end
    a = Union{Int64, NTuple{3, Int32}}[1, Int32.((2, 3, 4)), 5, Int32.((6, 7, 8))]
    b = CLArray(a)
    @opencl global_size=4 local_size=4 reverse_local!(b)
    @test Array(b) == reverse(a)

    # the elements are padded to 16 bytes, followed by a selector byte each. writing past
    # an undersized allocation goes unnoticed on the devices we test with, so check its size
    ir = sprint(io -> @device_code_llvm io=io dump_module=true @opencl launch=false reverse_local!(b))
    @test occursin(r"@local_memory = .* global \[68 x i8\]", ir)
end

@testset "array operations" begin
    x = CLArray(Union{Missing, Bool}[missing, true, false])
    @test isequal(Array(x .| false), [missing, true, false])
    @test isequal(Array(map(!, x)), [missing, false, true])
    @test any(x) === true
    @test all(x) === false
    @test isequal(Array(x[2:3]), [true, false])
    @test isequal(Array(reverse(x)), [false, true, missing])
    @test isequal(Array(vcat(x, x)), [missing, true, false, missing, true, false])

    y = CLArray(Union{Int, Nothing}[1, nothing, 3, nothing])
    @test sum(v -> something(v, 0), y) == 4
    @test mapreduce(v -> something(v, 10), max, y) == 10
    z = reshape(y, 2, 2)
    @test Array(z) == [1 3; nothing nothing]
    @test Array(map(v -> something(v, 0), z)) == [1 3; 0 0]

    fill!(y, nothing)
    @test all(isnothing, Array(y))
    fill!(y, 2)
    @test Array(y) == [2, 2, 2, 2]
end

@testset "scalar indexing ($M)" for M in union_memtypes
    a = CLVector{Union{Int, Nothing}, M}([1, nothing, 3])
    OpenCL.@allowscalar begin
        @test a[1] == 1
        @test a[2] === nothing
        a[1] = nothing
        a[2] = 2
        @test a[1] === nothing
        @test a[2] == 2
    end
    @test Array(a) == [nothing, 2, 3]
end

# host-accessible memory is indexed directly, also for unions
const union_host_memtypes =
    intersect(union_memtypes, [cl.UnifiedHostMemory, cl.UnifiedSharedMemory, cl.SharedVirtualMemory])
@testset "scalar indexing without allowscalar ($M)" for M in union_host_memtypes
    a = CLVector{Union{Int, Nothing}, M}([1, nothing, 3])
    @test a[1] == 1
    @test a[2] === nothing
    a[1] = nothing
    a[2] = 2
    @test a[1] === nothing
    @test a[2] == 2
    @test Array(a) == [nothing, 2, 3]

    # the element and its selector are read after outstanding work
    if M in union_kernel_memtypes
        @opencl global_size=3 bump!(a)
        @test a[2] == 3
    end

    c = view(CLVector{Union{Nothing, Missing}, M}([nothing, nothing, missing]), 2:3)
    @test c[2] === missing
    c[1] = missing
    @test isequal(Array(c), [missing, missing])
end

# resizing host memory isn't supported (it has no device address to copy to)
@testset "resize! ($M)" for M in filter(!=(cl.UnifiedHostMemory), union_memtypes)
    a = CLVector{Union{Int, Nothing}, M}([1, nothing, 3])
    resize!(a, 5)
    @test length(a) == 5
    @test Array(a)[1:3] == [1, nothing, 3]
    resize!(a, 2)
    @test Array(a) == [1, nothing]

    # a view keeps its selector bytes at an offset into its parent's memory
    b = CLVector{Union{Int, Nothing}, M}([nothing, nothing, 1, nothing, 3])
    v = view(b, 3:5)
    resize!(v, 4)
    @test Array(v)[1:3] == [1, nothing, 3]
    @test Array(b) == [nothing, nothing, 1, nothing, 3]

    c = CLVector{Union{Nothing, Missing}, M}([nothing, nothing, missing, missing])
    w = view(c, 3:4)
    resize!(w, 3)
    @test isequal(Array(w)[1:2], [missing, missing])
    resize!(w, 1)
    @test isequal(Array(w), [missing])
end

@testset "unsupported operations" begin
    a = Union{Int, Nothing}[1, nothing]
    # wrapped memory has no room for the selector bytes
    @test_throws ArgumentError unsafe_wrap(CLArray, a)
    @test_throws ArgumentError unsafe_wrap(CLArray, pointer(a), 2)
    for M in union_host_memtypes
        @test_throws ArgumentError unsafe_wrap(Array, CLVector{eltype(a), M}(a))
    end
    @test_throws "not all types are bitstypes" reinterpret(Float64, CLArray(a))
end
