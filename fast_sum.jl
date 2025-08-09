using OpenCL, pocl_jll, BenchmarkTools

using SPIRVIntrinsics, Atomix

function sum_columns_subgroup(X, result, M, N)
    col = get_global_id(1)
    row_thread = get_global_id(2)
    row_stride = get_global_size(2)

    if col > N
        return
    end

    partial = 0.0f0
    for row = row_thread:row_stride:M
        idx = (col - 1) * M + row  # column-major layout
        partial += X[idx]
    end

    # Subgroup shuffle-based warp reduction
    lane = get_sub_group_local_id()
    width = get_sub_group_size()

    offset = 1
    while offset < width
        if lane >= offset
            other = sub_group_shuffle(partial, lane - offset)
            partial += other
        end
        offset <<= 1
    end

    # Only one thread writes result
    if lane == 1
       Atomix.@atomic result[col] += partial
    end
    nothing
end


X = OpenCL.rand(Float32, 1000, 1000)
out = OpenCL.zeros(Float32, 1000)
@benchmark begin
    @opencl local_size = (1, 64) global_size = (1000, 64) extensions = ["SPV_EXT_shader_atomic_float_add"] sum_columns_subgroup(X, out, 1000, 1000)
    OpenCL.synchronize(out)
end
