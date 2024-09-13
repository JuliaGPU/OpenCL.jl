# Synchronization Functions

export barrier

const cl_mem_fence_flags = UInt32
const CLK_LOCAL_MEM_FENCE = cl_mem_fence_flags(1)
const CLK_GLOBAL_MEM_FENCE = cl_mem_fence_flags(2)

#barrier(flags=0) = @builtin_ccall("barrier", Cvoid, (UInt32,), flags)
@device_function barrier(flags=0) = Base.llvmcall(("""
        declare void @_Z7barrierj(i32) #0
        define void @entry(i32 %0) #1 {
            call void @_Z7barrierj(i32 %0)
            ret void
        }
        attributes #0 = { convergent }
        attributes #1 = { alwaysinline }
        """, "entry"),
    Cvoid, Tuple{Int32}, convert(Int32, flags))
push!(opencl_builtins, "_Z7barrierj")
# TODO: add support for attributes to @builting_ccall/LLVM.@typed_ccall
