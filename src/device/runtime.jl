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

## intrinsics for adding and accessing additional kernel arguments

# The amount of local shared memory we need for storing RNG state is determined
# dynamically at kernel launch time, so needs to be passed as additional arguments
# to the kernel.
# We define intrinsics that get transformed into additional kernel arguments which
# then get propagated across function calls to the caller.

function additional_arg_intr(mod::LLVM.Module, T_state, name)
    state_intr = if haskey(functions(mod), "julia.opencl.$name")
        functions(mod)["julia.opencl.$name"]
    else
        LLVM.Function(mod, "julia.opencl.$name", LLVM.FunctionType(T_state))
    end
    push!(function_attributes(state_intr), EnumAttribute("readnone", 0))

    return state_intr
end

# run-time equivalent
function additional_arg_value(state, name)
    @dispose ctx=Context() begin
        T_state = convert(LLVMType, state)

        # create function
        llvm_f, _ = create_function(T_state)
        mod = LLVM.parent(llvm_f)

        # get intrinsic
        state_intr = additional_arg_intr(mod, T_state, name)
        state_intr_ft = function_type(state_intr)

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            val = call!(builder, state_intr_ft, state_intr, Value[], name)

            ret!(builder, val)
        end

        call_function(llvm_f, state)
    end
end

for name in [:random_keys, :random_counters]
    @eval @inline @generated $name() =
        additional_arg_value(LLVMPtr{UInt32, AS.Workgroup}, $(String(name)))
end
