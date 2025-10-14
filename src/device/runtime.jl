# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

## exception handling

struct ExceptionInfo_st
    # whether an exception has been encountered (0 -> 1)
    status::Int32

    ExceptionInfo_st() = new(0)
end

# to simplify use of this struct, which is passed by-reference, use property overloading
const ExceptionInfo = Ptr{ExceptionInfo_st}
@inline function Base.getproperty(info::ExceptionInfo, sym::Symbol)
    if sym === :status
        unsafe_load(convert(Ptr{Int32}, info))
    else
        getfield(info, sym)
    end
end
@inline function Base.setproperty!(info::ExceptionInfo, sym::Symbol, value)
    if sym === :status
        unsafe_store!(convert(Ptr{Int32}, info), value)
    else
        setfield!(info, sym, value)
    end
end

## kernel state

struct KernelState
    exception_info::ExceptionInfo

    # XXX: Intel's SPIR-V compiler does not support array-valued kernel arguments, and Julia
    #      emits homogeneous structs as arrays. Work around this by including a dummy field.
    dummy::UInt32
end
KernelState(exception_info::ExceptionInfo) = KernelState(exception_info, 42)

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)

function signal_exception()
    info = kernel_state().exception_info

    # inform the host
    if info != C_NULL
        info.status = 1
        write_mem_fence(OpenCL.GLOBAL_MEM_FENCE)
    end

    return
end

malloc(sz) = C_NULL

report_oom(sz) = return

report_exception(ex) = return

report_exception_name(ex) = return

report_exception_frame(idx, func, file, line) = return
