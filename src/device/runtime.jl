# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

signal_exception() = return

malloc(sz) = C_NULL

report_oom(sz) = return

report_exception(ex) = return

report_exception_name(ex) = return

report_exception_frame(idx, func, file, line) = return

## kernel state

struct KernelState
    random_seed::UInt32
end

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)
@inline @generated random_keys() = GPUCompiler.call_custom_intrinsic(LLVMPtr{UInt32, AS.Workgroup}, "julia.spirv.random_keys", "random_keys")
@inline @generated random_counters() = GPUCompiler.call_custom_intrinsic(LLVMPtr{UInt32, AS.Workgroup}, "julia.spirv.random_counters", "random_counters")
