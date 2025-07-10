# Synchronization Functions

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
    const None = Relaxed            = 0x0000
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

@device_function control_barrier(execution_scope, memory_scope, memory_semantics) =
    Base.llvmcall(("""
        declare void @_Z22__spirv_ControlBarrierjjj(i32, i32, i32) #0
        define void @entry(i32 %execution, i32 %memory, i32 %semantics) #1 {
            call void @_Z22__spirv_ControlBarrierjjj(i32 %execution, i32 %memory, i32 %semantics)
            ret void
        }
        attributes #0 = { convergent }
        attributes #1 = { alwaysinline }
        """, "entry"),
    Cvoid, Tuple{Int32, Int32, Int32}, convert(Int32, execution_scope), convert(Int32, memory_scope), convert(Int32, memory_semantics))


## OpenCL API

export barrier, work_group_barrier

@enum memory_scope begin
    memory_scope_work_item
    memory_scope_sub_group
    memory_scope_work_group
    memory_scope_device
    memory_scope_all_svm_devices
    memory_scope_all_devices
end

function cl_scope_to_spirv(scope::memory_scope)
    if scope == memory_scope_work_item
        return Scope.Invocation
    elseif scope == memory_scope_sub_group
        return Scope.Subgroup
    elseif scope == memory_scope_work_group
        return Scope.Workgroup
    elseif scope == memory_scope_device
        return Scope.Device
    elseif scope == memory_scope_all_svm_devices || scope == memory_scope_all_devices
        return Scope.CrossDevice
    else
        error("Invalid memory scope: $scope")
    end
end

const cl_mem_fence_flags = UInt32
const CLK_LOCAL_MEM_FENCE = cl_mem_fence_flags(1)
const CLK_GLOBAL_MEM_FENCE = cl_mem_fence_flags(2)

function mem_fence_flags_to_semantics(flags)
    semantics = MemorySemantics.SequentiallyConsistent
    if (flags & CLK_LOCAL_MEM_FENCE) == CLK_LOCAL_MEM_FENCE
        semantics |= MemorySemantics.WorkgroupMemory
    end
    if (flags & CLK_GLOBAL_MEM_FENCE) == CLK_GLOBAL_MEM_FENCE
        semantics |= MemorySemantics.CrossWorkgroupMemory
    end
    return semantics
end

work_group_barrier(flags = 0, scope = memory_scope_work_group) =
    control_barrier(Scope.Workgroup, cl_scope_to_spirv(scope), mem_fence_flags_to_semantics(flags))

const barrier = work_group_barrier
