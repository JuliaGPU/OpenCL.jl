# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

signal_exception() = return

malloc(sz) = C_NULL

report_oom(sz) = return

report_exception(ex) = return

report_exception_name(ex) = return

report_exception_frame(idx, func, file, line) = return
