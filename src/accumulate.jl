# Use a smaller block size to work around a scan correctness issue
# with the Blelloch parallel prefix sum at larger block sizes (>=128).
const _ACCUMULATE_BLOCK_SIZE = 64

Base.accumulate!(
    op, B::CLArray, A::CLArray; init = zero(eltype(A)),
    block_size = _ACCUMULATE_BLOCK_SIZE, kwargs...
) =
    AK.accumulate!(op, B, A, OpenCLBackend(); init, block_size, kwargs...)

Base.accumulate(
    op, A::CLArray; init = zero(eltype(A)),
    block_size = _ACCUMULATE_BLOCK_SIZE, kwargs...
) =
    AK.accumulate(op, A, OpenCLBackend(); init, block_size, kwargs...)

Base.cumsum(src::CLArray; block_size = _ACCUMULATE_BLOCK_SIZE, kwargs...) =
    AK.cumsum(src, OpenCLBackend(); block_size, kwargs...)

Base.cumprod(src::CLArray; block_size = _ACCUMULATE_BLOCK_SIZE, kwargs...) =
    AK.cumprod(src, OpenCLBackend(); block_size, kwargs...)
