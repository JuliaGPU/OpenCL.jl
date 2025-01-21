@testset "Buffer" begin
    # simple buffer
    let buf = cl.Buffer{Int}(1)
        @test ndims(buf) == 1
        @test eltype(buf) == Int
        @test length(buf) == 1
        @test sizeof(buf) == sizeof(Int)
    end

    # memory copy
    let buf = cl.Buffer{Int}(1)
        src = [42]
        cl.enqueue_write(buf, pointer(src), sizeof(src); blocking=true)

        dst = [0]
        cl.enqueue_read(pointer(dst), buf, sizeof(dst); blocking=true)
        @test dst == [42]
    end

    # host accessible, mapped
    let buf = cl.Buffer{Int}(1; host_accessible=true)
        src = [42]
        cl.enqueue_write(buf, pointer(src), sizeof(src); blocking=true)

        ptr, evt = cl.enqueue_map(buf, sizeof(buf), :rw)
        wait(evt)
        mapped = unsafe_wrap(Array, convert(Ptr{Int}, ptr), 1; own=false)
        @test mapped[] == 42
        cl.enqueue_unmap(buf, ptr) |> wait
    end

    # re-use host buffer, without copy
    let arr = [1,2,3]
        buf = cl.Buffer(arr; copy=false)

        dst = similar(arr)
        cl.enqueue_read(pointer(dst), buf, sizeof(dst); blocking=true)
        @test dst == arr

        # we still need to map, despite copy=false
        ptr, evt = cl.enqueue_map(buf, sizeof(buf), :rw)
        wait(evt)
        mapped_arr = unsafe_wrap(Array, convert(Ptr{Int}, ptr), 3; own=false)
        mapped_arr .= 42
        cl.enqueue_unmap(buf, ptr) |> wait

        # but our pre-allocated buffer should have been updated too
        @test arr == [42,42,42]

        # and we can read it back
        cl.enqueue_read(pointer(dst), buf, sizeof(dst); blocking=true)
        @test dst == arr
    end

    # re-use host buffer, but copy
    let arr = [1,2,3]
        buf = cl.Buffer(arr; copy=true)

        dst = similar(arr)
        cl.enqueue_read(pointer(dst), buf, sizeof(dst); blocking=true)
        @test dst == arr

        arr .= 42

        # but our pre-allocated buffer should not have been updated
        cl.enqueue_read(pointer(dst), buf, sizeof(dst); blocking=true)
        @test dst == [1,2,3]
    end

    # fill
    let buf = cl.Buffer{Int}(3)
        cl.enqueue_fill(buf, 42, 3)

        arr = Vector{Int}(undef, 3)
        cl.enqueue_read(pointer(arr), buf, sizeof(arr); blocking=true)
        @test arr == [42,42,42]
    end
end


@testset "SVM Buffer" begin
    # simple buffer
    let buf = cl.svm_alloc(cl.context(), sizeof(Int))
        @test sizeof(buf) == sizeof(Int)
    end

    # memory copy
    let buf = cl.svm_alloc(cl.context(), sizeof(Int))
        ptr = pointer(buf)

        src = [42]
        cl.enqueue_svm_copy(ptr, pointer(src), sizeof(src))

        dst = [0]
        cl.enqueue_svm_copy(pointer(dst), ptr, sizeof(dst); blocking = true)
        @test dst == [42]
    end

    # memory map

    let buf = cl.svm_alloc(cl.context(), sizeof(Int))
        ptr = pointer(buf)

        src = [42]
        cl.enqueue_svm_copy(ptr, pointer(src), sizeof(src))

        evt = cl.enqueue_svm_map(ptr, sizeof(src), :rw)
        wait(evt)
        mapped = unsafe_wrap(Array, Ptr{Int}(UInt(ptr)), 1; own = false)
        @test mapped[] == 42
        mapped[] = 100
        cl.enqueue_svm_unmap(ptr) |> cl.wait

        dst = [0]
        cl.enqueue_svm_copy(pointer(dst), ptr, sizeof(dst); blocking = true)
        @test dst == [100]
    end

    # fill
    let buf = cl.svm_alloc(cl.context(), 3 * sizeof(Int))
        ptr = pointer(buf)

        cl.enqueue_svm_fill(ptr, pointer([42]), sizeof(Int), 3 * sizeof(Int))

        dst = Vector{Int}(undef, 3)
        cl.enqueue_svm_copy(pointer(dst), ptr, sizeof(dst); blocking = true)
        @test dst == [42,42,42]
    end
end
