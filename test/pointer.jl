using OpenCL.cl

# constructors
voidptr_a = CLPtr{Cvoid}(Int(0xDEADBEEF))
@test reinterpret(Ptr{Cvoid}, voidptr_a) == Ptr{Cvoid}(Int(0xDEADBEEF))

# getters
@test eltype(voidptr_a) == Cvoid

# comparisons
voidptr_b = CLPtr{Cvoid}(Int(0xCAFEBABE))
@test voidptr_a != voidptr_b


@testset "conversions" begin

    # between host and device pointers
    @test_throws ArgumentError convert(Ptr{Cvoid}, voidptr_a)

    # between device pointers
    intptr_a = CLPtr{Int}(Int(0xDEADBEEF))
    @test convert(typeof(intptr_a), voidptr_a) == intptr_a

    # convert back and forth from UInt
    intptr_b = CLPtr{Int}(Int(0xDEADBEEF))
    @test convert(UInt, intptr_b) == 0xDEADBEEF
    @test convert(CLPtr{Int}, Int(0xDEADBEEF)) == intptr_b
    @test Int(intptr_b) == Int(0xDEADBEEF)

    # pointer arithmetic
    intptr_c = CLPtr{Int}(Int(0xDEADBEEF))
    intptr_d = 2 + intptr_c
    @test isless(intptr_c, intptr_d)
    @test intptr_d - intptr_c == 2
    @test intptr_d - 2 == intptr_c
end


@testset "GPU or CPU integration" begin
    a = [1]
    ccall(:clock, Nothing, (Ptr{Int},), a)
    @test_throws Exception ccall(:clock, Nothing, (CLPtr{Int},), a)
    ccall(:clock, Nothing, (PtrOrCLPtr{Int},), a)

    b = CLArray{eltype(a), ndims(a), cl.Buffer}(undef, size(a))
    @test device_accessible(b)
    ccall(:clock, Nothing, (CLPtr{Int},), b)
    @test !host_accessible(b)
    @test_throws Exception ccall(:clock, Nothing, (Ptr{Int},), b)
    ccall(:clock, Nothing, (PtrOrCLPtr{Int},), b)
end


@testset "reference values" begin
    # Ref

    @test typeof(Base.cconvert(Ref{Int}, 1)) == typeof(Ref(1))
    @test Base.unsafe_convert(Ref{Int}, Base.cconvert(Ref{Int}, 1)) isa Ptr{Int}

    ptr = reinterpret(Ptr{Int}, C_NULL)
    @test Base.cconvert(Ref{Int}, ptr) == ptr
    @test Base.unsafe_convert(Ref{Int}, Base.cconvert(Ref{Int}, ptr)) == ptr

    arr = [1]
    @test Base.cconvert(Ref{Int}, arr) isa Base.RefArray{Int, typeof(arr)}
    @test Base.unsafe_convert(Ref{Int}, Base.cconvert(Ref{Int}, arr)) == pointer(arr)


    # CLRef

    @test typeof(Base.cconvert(CLRef{Int}, 1)) == typeof(CLRef(1))
    @test Base.unsafe_convert(CLRef{Int}, Base.cconvert(CLRef{Int}, 1)) isa CLRef{Int}

    clptr = reinterpret(CLPtr{Int}, C_NULL)
    @test Base.cconvert(CLRef{Int}, clptr) == clptr
    @test Base.unsafe_convert(CLRef{Int}, Base.cconvert(CLRef{Int}, clptr)) == Base.bitcast(CLRef{Int}, clptr)

    clarr = OpenCL.CLArray([1])
    @test Base.cconvert(CLRef{Int}, clarr) isa cl.CLRefArray{Int, typeof(clarr)}
    @test Base.unsafe_convert(CLRef{Int}, Base.cconvert(CLRef{Int}, clarr)) == Base.bitcast(CLRef{Int}, pointer(clarr))


    # RefOrCLRef

    @test typeof(Base.cconvert(RefOrCLRef{Int}, 1)) == typeof(Ref(1))
    @test Base.unsafe_convert(RefOrCLRef{Int}, Base.cconvert(RefOrCLRef{Int}, 1)) isa RefOrCLRef{Int}

    @test Base.cconvert(RefOrCLRef{Int}, ptr) == ptr
    @test Base.unsafe_convert(RefOrCLRef{Int}, Base.cconvert(RefOrCLRef{Int}, ptr)) == Base.bitcast(RefOrCLRef{Int}, ptr)

    @test Base.cconvert(RefOrCLRef{Int}, clptr) == clptr
    @test Base.unsafe_convert(RefOrCLRef{Int}, Base.cconvert(RefOrCLRef{Int}, clptr)) == Base.bitcast(RefOrCLRef{Int}, clptr)

    @test Base.cconvert(RefOrCLRef{Int}, arr) isa Base.RefArray{Int, typeof(arr)}
    @test Base.unsafe_convert(RefOrCLRef{Int}, Base.cconvert(RefOrCLRef{Int}, arr)) == Base.bitcast(RefOrCLRef{Int}, pointer(arr))

    @test Base.cconvert(RefOrCLRef{Int}, clarr) isa cl.CLRefArray{Int, typeof(clarr)}
    @test Base.unsafe_convert(RefOrCLRef{Int}, Base.cconvert(RefOrCLRef{Int}, clarr)) == Base.bitcast(RefOrCLRef{Int}, pointer(clarr))
end

