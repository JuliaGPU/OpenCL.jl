## exception mailbox

# Kernels report into host-visible memory passed through KernelState. The host reads and
# resets the mailbox after all queues that used it have completed.

struct ExceptionInfo_st
    # set by the first work-item that throws (0 -> 1), which then records the rest
    status::Int32
    # the throw site, indexing the table of the kernel that threw (see
    # `GPUCompiler.uses_exception_sites`); 0 when nothing is known about it
    site::Int32
    # the kernel's table of throw sites, as numbered for the launch (see `KernelState`)
    sites::UInt32
    # Keep positions at the builtins' 64-bit width and pad to four components. Narrowing
    # or merging adjacent three-component accesses can produce illegal six-wide SPIR-V
    # vectors.
    work_item::NTuple{4, UInt64}
    work_group::NTuple{4, UInt64}

    ExceptionInfo_st() = new(0, 0, 0, (0, 0, 0, 0), (0, 0, 0, 0))
end

# Access fields through typed device pointers derived from the host struct layout.
const ExceptionInfo = LLVMPtr{UInt8, AS.CrossWorkgroup}

const EXCEPTION_INFO_OFFSETS = NamedTuple{fieldnames(ExceptionInfo_st)}(
    Tuple(Int(fieldoffset(ExceptionInfo_st, i)) for i in 1:fieldcount(ExceptionInfo_st)))

@inline info_field(info::ExceptionInfo, ::Val{field}) where {field} =
    reinterpret(LLVMPtr{fieldtype(ExceptionInfo_st, field), AS.CrossWorkgroup},
                info + getfield(EXCEPTION_INFO_OFFSETS, field))

# the mailbox address as carried by the kernel state; null when the device has no
# host-accessible memory to put a mailbox in (see `allocate_exception_mailbox`).
@device_function @inline exception_info() = reinterpret(ExceptionInfo, kernel_state().exception_info)
@inline has_exception_info(info::ExceptionInfo) = info != reinterpret(ExceptionInfo, 0)

# Claims must cover all work-groups and queues on the device. SPIRVIntrinsics' public
# atomic helpers are relaxed, so use UnsafeAtomics' primitives with the orderings needed here.
#
# Claim with an atomic max (0 -> 1) rather than a compare-and-swap, which PoCL implements
# with a builtin that returns the old value through a stack slot, which PoCL's work-item
# loops can widen into an array over the work-group (JuliaGPU/KernelAbstractions.jl#974).
@inline atomic_claim_device!(p::LLVMPtr{Int32, AS.CrossWorkgroup}) =
    UnsafeAtomics.Internal.llvm_rmw!(p, Val(:max), 1 % Int32, Val(:acq_rel), Val(:device),
                                     Val(false), Val(4), Val(()))

# throw a device-side exception reported as `name` with `reason`
using GPUCompiler: @gputhrow

# The work-item that threw stops executing after this. The first one to throw records where
# and which work-item it was; the host knows what was thrown there. Kernel completion makes
# the record visible to the host.
#
# Nothing reachable from here may throw, not even a checked conversion like `Int32(x)`.
function exception_site(site::Int32)
    info = exception_info()
    if has_exception_info(info) && atomic_claim_device!(info_field(info, Val(:status))) == 0
        unsafe_store!(info_field(info, Val(:site)), site)
        unsafe_store!(info_field(info, Val(:sites)), kernel_state().exception_sites)
        unsafe_store!(info_field(info, Val(:work_item)),
                      (get_local_id(1) % UInt64, get_local_id(2) % UInt64,
                       get_local_id(3) % UInt64, 0 % UInt64))
        unsafe_store!(info_field(info, Val(:work_group)),
                      (get_group_id(1) % UInt64, get_group_id(2) % UInt64,
                       get_group_id(3) % UInt64, 0 % UInt64))
    end
    return
end

report_oom(sz) = return

malloc(sz) = C_NULL

## kernel state

struct KernelState
    random_seed::UInt32
    # the bytes of dynamic local memory that the launch asked for (see
    # `dynamic_localmemory`); in the padding of the seed
    dynamic_localmemory_size::UInt32

    # address of the device-side exception mailbox (`ExceptionInfo_st`); zero when the
    # device has no host-accessible memory to put it in. carried as an integer rather than
    # a pointer: SPIR-V validation rejects pointer-typed fields in by-value kernel arguments.
    exception_info::UInt64
    # the table of throw sites of the launched kernel, numbered per mailbox (see
    # `exception_sites_id`); 0 when the kernel has none
    exception_sites::UInt32
end

@device_function @inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)

## intrinsics for adding and accessing additional kernel arguments

# The amount of local shared memory we need for storing RNG state is determined
# dynamically at kernel launch time, so needs to be passed as additional arguments
# to the kernel.
# We define intrinsics that get transformed into additional kernel arguments which
# then get propagated across function calls to the caller.

function additional_arg_intr(mod::LLVM.Module, T_state, name)
    get!(mod.functions, "julia.opencl.$name") do
        state_intr = LLVM.Function(mod, "julia.opencl.$name", LLVM.FunctionType(T_state))
        state_intr.memory_effects = MemoryEffects(:none)
        state_intr
    end
end

# run-time equivalent
@llvmgenerated builder function additional_arg_value(::Type{T}, ::Val{name})::T where {T,name}
    state_intr = additional_arg_intr(current_module(builder), convert(LLVMType, T), name)
    call!(builder, state_intr.function_type, state_intr, Value[], String(name))
end

for name in [:random_keys, :random_counters]
    @eval @device_function @inline $name() = additional_arg_value(LLVMPtr{UInt32, AS.Workgroup}, Val($(QuoteNode(name))))
end

# The block of local memory whose size is chosen at launch (see `KI.dynamic_localmemory`):
# another local memory argument, which every launch of a kernel that uses it has to size
# (see `dynamic_local_bytes`). `kernel_state().dynamic_localmemory_size` is how much of it
# the launch asked for.
#
# OpenCL doesn't specify the alignment of the memory of such an argument, so the launch
# allocates `dynamic_localmemory_alignment` bytes more than it was asked for, and the block
# starts at the first aligned address.
const dynamic_localmemory_alignment = 16

@device_function @inline function dynamic_localmemory()
    ptr = additional_arg_value(LLVMPtr{UInt8, AS.Workgroup}, Val(:dynamic_localmemory))
    return ptr + (-UInt(ptr) & UInt(dynamic_localmemory_alignment - 1))
end
