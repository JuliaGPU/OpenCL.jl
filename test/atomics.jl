@testset "atomics" begin

function atomic_count(counter)
    OpenCL.@atomic counter[] += 1
    return
end

@testset "atomic_add! ($T)" for T in [Int32, UInt32, Int64, UInt64]
    if sizeof(T) == 4 || "cl_khr_int64_extended_atomics" in cl.device().extensions
        a = OpenCL.zeros(T)
        @opencl global_size=1000 atomic_count(a)
        @test OpenCL.@allowscalar a[] == 1000
    end
end

end
