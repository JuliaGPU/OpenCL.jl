abstract type CLBackend end

struct SVMBackend <: CLBackend end

struct USMBackend <: CLBackend end

struct MemBackend <: CLBackend end

function select_backend(dev::Device = cl.device())
    return @memoize begin
        usmcapabilities = usm_capabilities(dev)
        if isnothing(usmcapabilities) || !(usmcapabilities.usm_host_capabilities.usm_access && usmcapabilities.usm_device_capabilities.usm_access)
            return SVMBackend
        else
            return USMBackend
        end
    end::Type{<:CLBackend}
end

function select_buffer(dev::Device = cl.device())
    return @memoize begin
        backend = select_backend(dev)
        if backend == USMBackend
            UnifiedDeviceMemory
        else
            SharedVirtualMemory
        end
    end::Type{<:AbstractMemory}
end

function get_backend_from_buffer(x::Type{<:AbstractMemory})
    return if x == SharedVirtualMemory
        SVMBackend
    else
        USMBackend
    end
end

function abstract_kernel_exec_info_ptrs(backend::Type{<:CLBackend})
    return if backend == SVMBackend
        CL_KERNEL_EXEC_INFO_SVM_PTRS
    else
        CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL
    end
end

function set_kernel_arg_abstract_pointer(backend::Type{<:CLBackend})
    return if backend == SVMBackend
        ext_clSetKernelArgSVMPointer
    else
        ext_clSetKernelArgMemPointerINTEL
    end
end

function set_kernel_arg_abstract_pointer(backend::Type{<:AbstractMemory})
    return if backend == SharedVirtualMemory
        ext_clSetKernelArgSVMPointer
    else
        ext_clSetKernelArgMemPointerINTEL
    end
end
