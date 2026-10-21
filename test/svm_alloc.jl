@testset "SVM allocation failure handling" begin
    sentinel = cl.CLPtr{Cvoid}(UInt(0x1000))
    flags = cl.CL_MEM_READ_WRITE
    attempts = Ref(0)
    immediate = () -> (attempts[] += 1; sentinel)
    @test cl.checked_svm_alloc(immediate, 36, 4, flags) == sentinel
    @test attempts[] == 1

    attempts[] = 0
    recover = () -> (attempts[] += 1; attempts[] == 1 ? cl.CL_NULL : sentinel)
    @test cl.checked_svm_alloc(recover, 36, 4, flags) == sentinel
    @test attempts[] == 2

    attempts[] = 0
    fail = () -> (attempts[] += 1; cl.CL_NULL)
    @test_throws r"clSVMAlloc returned NULL after reclamation" cl.checked_svm_alloc(fail, 36, 4, flags)
    @test attempts[] == 3
    for bytes in (0, -1, big(typemax(Int)) + 1)
        @test_throws ArgumentError cl.svm_alloc(bytes)
    end
    for alignment in (-1, 3, big(typemax(cl.cl_uint)) + 1)
        @test_throws ArgumentError cl.svm_alloc(36; alignment)
    end
    if cl.SVMBackend() in cl.supported_memory_backends(cl.device())
        ctx = cl.context()
        GC.@preserve ctx begin
            before = ctx.reference_count
            allocation = cl.svm_alloc(36; alignment=4)
            @test pointer(allocation) != cl.CL_NULL
            @test ctx.reference_count == before + 1
            cl.svm_free(allocation)
            @test ctx.reference_count == before
        end
    end
end
