@testset "Buffer" begin
    # simple buffer
    let buf = cl.Buffer{Int}(1)
        @test ndims(buf) == 1
        @test eltype(buf) == Int
        @test length(buf) == 1
        @test sizeof(buf) == sizeof(Int)
    end

    # host accessible, mapped
    let buf = cl.Buffer{Int}(1; host_accessible=true)
        unsafe_copyto!(buf, [42], 1; blocking=true)

        arr, evt = cl.unsafe_map!(buf, (1,), :rw)
        wait(evt)
        @test arr[] == 42
        cl.unsafe_unmap!(buf, arr)
    end

    # re-use host buffer, without copy
    let arr = [1,2,3]
        buf = cl.Buffer(arr; copy=false)

        dst = similar(arr)
        unsafe_copyto!(dst, buf, 3; blocking=true)
        @test dst == arr

        # we still need to map, despite copy=false
        mapped_arr, evt = cl.unsafe_map!(buf, (3,), :rw)
        wait(evt)
        mapped_arr .= 42
        cl.unsafe_unmap!(buf, mapped_arr) |> wait

        # but our pre-allocated buffer should have been updated too
        @test arr == [42,42,42]

        unsafe_copyto!(dst, buf, 3; blocking=true)
        @test dst == arr
    end

    # re-use host buffer, but copy
    let arr = [1,2,3]
        buf = cl.Buffer(arr; copy=true)

        dst = similar(arr)
        unsafe_copyto!(dst, buf, 3; blocking=true)
        @test dst == arr

        arr .= 42

        unsafe_copyto!(dst, buf, 3; blocking=true)
        @test dst == [1,2,3]
    end

    # fill
    let buf = cl.Buffer{Int}(3)
        cl.unsafe_fill!(buf, 42, 3)
        arr = Vector{Int}(undef, 3)
        unsafe_copyto!(arr, buf, 3; blocking=true)
        @test arr == [42,42,42]
    end
end
