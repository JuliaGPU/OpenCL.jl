# Sub-group collectives from `cl_khr_subgroups`, which the SPIR-V back-end lowers to the
# `OpGroup*` instructions. All work-items of the sub-group have to execute them together.

const collective_types = [Int32, UInt32, Int64, UInt64, Float16, Float32, Float64]

for op in (:add, :min, :max)
    reduce = Symbol(:sub_group_reduce_, op)
    scan_inclusive = Symbol(:sub_group_scan_inclusive_, op)
    scan_exclusive = Symbol(:sub_group_scan_exclusive_, op)
    @eval export $reduce, $scan_inclusive, $scan_exclusive
    for T in collective_types
        @eval begin
            @device_function $reduce(x::$T) =
                @builtin_ccall($(String(reduce)), $T, ($T,), x, convergent = true)
            @device_function $scan_inclusive(x::$T) =
                @builtin_ccall($(String(scan_inclusive)), $T, ($T,), x, convergent = true)
            @device_function $scan_exclusive(x::$T) =
                @builtin_ccall($(String(scan_exclusive)), $T, ($T,), x, convergent = true)
        end
    end
end

# `x` of the work-item with (1-based) sub-group local id `lane`, which has to be the same for
# all work-items of the sub-group
export sub_group_broadcast
for T in collective_types
    @eval @device_function sub_group_broadcast(x::$T, lane::Integer) =
        @builtin_ccall("sub_group_broadcast", $T, ($T, UInt32), x, (lane - 1) % UInt32, convergent = true)
end
