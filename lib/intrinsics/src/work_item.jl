# Work-Item Functions

export get_work_dim,
       get_global_size, get_global_id,
       get_local_size, get_enqueued_local_size, get_local_id,
       get_num_groups, get_group_id,
       get_global_offset,
       get_global_linear_id, get_local_linear_id

# NOTE: these functions now unsafely truncate to Int to avoid top bit checks.
#       we should probably use range metadata instead.

@device_function get_work_dim() = @builtin_ccall("get_work_dim", UInt32, ()) % Int

@device_function get_global_size(dimindx::Integer = 1u32) = @builtin_ccall("get_global_size", UInt, (UInt32,), dimindx - 1u32) % Int
@device_function get_global_id(dimindx::Integer = 1u32) = @builtin_ccall("get_global_id", UInt, (UInt32,), dimindx - 1u32) % Int + 1

@device_function get_local_size(dimindx::Integer = 1u32) = @builtin_ccall("get_local_size", UInt, (UInt32,), dimindx - 1u32) % Int
@device_function get_enqueued_local_size(dimindx::Integer = 1u32) = @builtin_ccall("get_enqueued_local_size", UInt, (UInt32,), dimindx - 1) % Int
@device_function get_local_id(dimindx::Integer = 1u32) = @builtin_ccall("get_local_id", UInt, (UInt32,), dimindx - 1u32) % Int + 1

@device_function get_num_groups(dimindx::Integer = 1u32) = @builtin_ccall("get_num_groups", UInt, (UInt32,), dimindx - 1u32) % Int
@device_function get_group_id(dimindx::Integer = 1u32) = @builtin_ccall("get_group_id", UInt, (UInt32,), dimindx - 1u32) % Int + 1

@device_function get_global_offset(dimindx::Integer = 1u32) = @builtin_ccall("get_global_offset", UInt, (UInt32,), dimindx - 1u32) % Int + 1

@device_function get_global_linear_id() = @builtin_ccall("get_global_linear_id", UInt, ()) % Int + 1
@device_function get_local_linear_id() = @builtin_ccall("get_local_linear_id", UInt, ()) % Int + 1
