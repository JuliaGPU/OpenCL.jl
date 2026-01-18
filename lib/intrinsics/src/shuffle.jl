export sub_group_shuffle

const gentypes = [Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float16, Float32, Float64]

for gentype in gentypes
    @eval begin
        @device_function sub_group_shuffle(x::$gentype, i::Integer) = @builtin_ccall("sub_group_shuffle", $gentype, ($gentype, Int32), x, i % Int32 - 1i32)
    end
end
