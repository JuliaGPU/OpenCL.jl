@testset "intrinsics" begin

@testset "barrier" begin

@on_device barrier(OpenCL.LOCAL_MEM_FENCE)
@on_device barrier(OpenCL.GLOBAL_MEM_FENCE)
@on_device barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)

@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE)
@on_device work_group_barrier(OpenCL.GLOBAL_MEM_FENCE)
@on_device work_group_barrier(OpenCL.IMAGE_MEM_FENCE)

@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE | OpenCL.IMAGE_MEM_FENCE)
@on_device work_group_barrier(OpenCL.GLOBAL_MEM_FENCE | OpenCL.LOCAL_MEM_FENCE | OpenCL.IMAGE_MEM_FENCE)

@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_work_item)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_work_group)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_device)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_all_svm_devices)
@on_device work_group_barrier(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_scope_sub_group)

end

@testset "mem_fence" begin

@on_device mem_fence(OpenCL.LOCAL_MEM_FENCE)
@on_device mem_fence(OpenCL.GLOBAL_MEM_FENCE)
@on_device mem_fence(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)

@on_device read_mem_fence(OpenCL.LOCAL_MEM_FENCE)
@on_device read_mem_fence(OpenCL.GLOBAL_MEM_FENCE)
@on_device read_mem_fence(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)

@on_device write_mem_fence(OpenCL.LOCAL_MEM_FENCE)
@on_device write_mem_fence(OpenCL.GLOBAL_MEM_FENCE)
@on_device write_mem_fence(OpenCL.LOCAL_MEM_FENCE | OpenCL.GLOBAL_MEM_FENCE)

end

@testset "atomic_work_item_fence" begin

@on_device atomic_work_item_fence(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_order_relaxed, OpenCL.memory_scope_work_item)
@on_device atomic_work_item_fence(OpenCL.GLOBAL_MEM_FENCE, OpenCL.memory_order_acquire, OpenCL.memory_scope_work_group)
@on_device atomic_work_item_fence(OpenCL.IMAGE_MEM_FENCE, OpenCL.memory_order_release, OpenCL.memory_scope_device)
@on_device atomic_work_item_fence(OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_order_acq_rel, OpenCL.memory_scope_all_svm_devices)
@on_device atomic_work_item_fence(OpenCL.GLOBAL_MEM_FENCE, OpenCL.memory_order_seq_cst, OpenCL.memory_scope_sub_group)
@on_device atomic_work_item_fence(OpenCL.IMAGE_MEM_FENCE | OpenCL.LOCAL_MEM_FENCE, OpenCL.memory_order_acquire, OpenCL.memory_scope_sub_group)

end

end

