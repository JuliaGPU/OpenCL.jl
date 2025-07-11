# Synchronization Functions

## SPIR-V wrappers

module Scope
    const CrossDevice  = 0
    const Device       = 1
    const Workgroup    = 2
    const Subgroup     = 3
    const Invocation   = 4
    const QueueFamily  = 5
    const ShaderCall   = 6
end

module MemorySemantics
    const None = const Relaxed      = 0x0000
    const Acquire                   = 0x0002
    const Release                   = 0x0004
    const AcquireRelease            = 0x0008
    const SequentiallyConsistent    = 0x0010
    const UniformMemory             = 0x0040
    const SubgroupMemory            = 0x0080
    const WorkgroupMemory           = 0x0100
    const CrossWorkgroupMemory      = 0x0200
    const AtomicCounterMemory       = 0x0400
    const ImageMemory               = 0x0800
    const OutputMemory              = 0x1000
    const MakeAvailable             = 0x2000
    const MakeVisible               = 0x4000
    const Signal                    = 0x8000
end

@device_function @inline memory_barrier(scope, semantics) =
    @builtin_ccall("__spirv_MemoryBarrier", Cvoid, (UInt32, UInt32), scope, semantics)

@device_function @inline control_barrier(execution_scope, memory_scope, memory_semantics) =
    @builtin_ccall("__spirv_ControlBarrier", Cvoid, (UInt32, UInt32, UInt32),
                   execution_scope, memory_scope, memory_semantics)


## OpenCL types

const cl_mem_fence_flags = UInt32
const LOCAL_MEM_FENCE = cl_mem_fence_flags(1)
const GLOBAL_MEM_FENCE = cl_mem_fence_flags(2)
const IMAGE_MEM_FENCE = cl_mem_fence_flags(4)

@inline function mem_fence_flags_to_semantics(flags)
    semantics = MemorySemantics.None
    if (flags & LOCAL_MEM_FENCE) == LOCAL_MEM_FENCE
        semantics |= MemorySemantics.WorkgroupMemory
    end
    if (flags & GLOBAL_MEM_FENCE) == GLOBAL_MEM_FENCE
        semantics |= MemorySemantics.CrossWorkgroupMemory
    end
    return semantics
end

@enum memory_scope begin
    memory_scope_work_item
    memory_scope_sub_group
    memory_scope_work_group
    memory_scope_device
    memory_scope_all_svm_devices
    memory_scope_all_devices
end

@inline function cl_scope_to_spirv(scope)
    if scope == memory_scope_work_item
        Scope.Invocation
    elseif scope == memory_scope_sub_group
        Scope.Subgroup
    elseif scope == memory_scope_work_group
        Scope.Workgroup
    elseif scope == memory_scope_device
        Scope.Device
    elseif scope == memory_scope_all_svm_devices || scope == memory_scope_all_devices
        Scope.CrossDevice
    else
        error("Invalid memory scope: $scope")
    end
end

@enum memory_order begin
    memory_order_relaxed
    memory_order_acquire
    memory_order_release
    memory_order_acq_rel
    memory_order_seq_cst
end


## OpenCL memory barriers

export atomic_work_item_fence, mem_fence, read_mem_fence, write_mem_fence

@inline function atomic_work_item_fence(flags, order, scope)
    semantics = mem_fence_flags_to_semantics(flags)
    if order == memory_order_relaxed
        memory_barrier(scope, semantics | MemorySemantics.Relaxed)
    elseif order == memory_order_acquire
        memory_barrier(scope, semantics | MemorySemantics.Acquire)
    elseif order == memory_order_release
        memory_barrier(scope, semantics | MemorySemantics.Release)
    elseif order == memory_order_acq_rel
        memory_barrier(scope, semantics | MemorySemantics.AcquireRelease)
    elseif order == memory_order_seq_cst
        memory_barrier(scope, semantics | MemorySemantics.SequentiallyConsistent)
    else
        error("Invalid memory order: $order")
    end
end

# legacy fence functions
mem_fence(flags) = atomic_work_item_fence(flags, memory_order_acq_rel, memory_scope_work_group)
read_mem_fence(flags) = atomic_work_item_fence(flags, memory_order_acquire, memory_scope_work_group)
write_mem_fence(flags) = atomic_work_item_fence(flags, memory_order_release, memory_scope_work_group)


## OpenCL execution barriers

export barrier, work_group_barrier

@inline work_group_barrier(flags, scope = memory_scope_work_group) =
    control_barrier(Scope.Workgroup, cl_scope_to_spirv(scope),
                    MemorySemantics.SequentiallyConsistent | mem_fence_flags_to_semantics(flags))

barrier(flags) = work_group_barrier(flags)
