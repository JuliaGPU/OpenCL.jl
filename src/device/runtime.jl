# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

signal_exception() = return

malloc(sz) = C_NULL

report_oom(sz) = return

function report_exception(ex)
    @printf(
        "ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d).\n",
        ex, get_global_id(UInt32(0)), get_global_id(UInt32(1)), get_global_id(UInt32(2))
    )
    return
end

function report_exception_name(ex)
    @printf(
        "ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d).\nStacktrace:\n",
        ex, get_global_id(UInt32(0)), get_global_id(UInt32(1)), get_global_id(UInt32(2))
    )
    return
end

function report_exception_frame(idx, func, file, line)
    @printf(" [%d] %s at %s:%d\n", idx, func, file, line)
    return
end
