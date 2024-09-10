# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(CLError(res))
end

function check(f)
    res = retry_reclaim(err -> err == CL_OUT_OF_RESOURCES ||
                                   err == CL_MEM_OBJECT_ALLOCATION_FAILURE ||
                                   err == CL_OUT_OF_HOST_MEMORY) do
        return f()
    end

    if res != CL_SUCCESS
        throw_api_error(res)
    end

    return
end

const __darwin_intptr_t = Clong

const intptr_t = __darwin_intptr_t

const cl_int = Int32

const cl_uint = UInt32

const cl_ulong = UInt64

const cl_GLuint = Cuint

const cl_GLint = Cint

const cl_GLenum = Cuint

mutable struct _cl_platform_id end

mutable struct _cl_device_id end

mutable struct _cl_context end

mutable struct _cl_command_queue end

mutable struct _cl_mem end

mutable struct _cl_program end

mutable struct _cl_kernel end

mutable struct _cl_event end

mutable struct _cl_sampler end

const cl_platform_id = Ptr{_cl_platform_id}

const cl_device_id = Ptr{_cl_device_id}

const cl_context = Ptr{_cl_context}

const cl_command_queue = Ptr{_cl_command_queue}

const cl_mem = Ptr{_cl_mem}

const cl_program = Ptr{_cl_program}

const cl_kernel = Ptr{_cl_kernel}

const cl_event = Ptr{_cl_event}

const cl_sampler = Ptr{_cl_sampler}

const cl_bool = cl_uint

const cl_bitfield = cl_ulong

const cl_properties = cl_ulong

const cl_device_type = cl_bitfield

const cl_platform_info = cl_uint

const cl_device_info = cl_uint

const cl_device_fp_config = cl_bitfield

const cl_device_mem_cache_type = cl_uint

const cl_device_local_mem_type = cl_uint

const cl_device_exec_capabilities = cl_bitfield

const cl_device_svm_capabilities = cl_bitfield

const cl_command_queue_properties = cl_bitfield

const cl_device_partition_property = intptr_t

const cl_device_affinity_domain = cl_bitfield

const cl_context_properties = intptr_t

const cl_context_info = cl_uint

const cl_queue_properties = cl_properties

const cl_command_queue_info = cl_uint

const cl_channel_order = cl_uint

const cl_channel_type = cl_uint

const cl_mem_flags = cl_bitfield

const cl_svm_mem_flags = cl_bitfield

const cl_mem_object_type = cl_uint

const cl_mem_info = cl_uint

const cl_mem_migration_flags = cl_bitfield

const cl_image_info = cl_uint

const cl_buffer_create_type = cl_uint

const cl_addressing_mode = cl_uint

const cl_filter_mode = cl_uint

const cl_sampler_info = cl_uint

const cl_map_flags = cl_bitfield

const cl_pipe_properties = intptr_t

const cl_pipe_info = cl_uint

const cl_program_info = cl_uint

const cl_program_build_info = cl_uint

const cl_program_binary_type = cl_uint

const cl_build_status = cl_int

const cl_kernel_info = cl_uint

const cl_kernel_arg_info = cl_uint

const cl_kernel_arg_address_qualifier = cl_uint

const cl_kernel_arg_access_qualifier = cl_uint

const cl_kernel_arg_type_qualifier = cl_bitfield

const cl_kernel_work_group_info = cl_uint

const cl_kernel_sub_group_info = cl_uint

const cl_event_info = cl_uint

const cl_command_type = cl_uint

const cl_profiling_info = cl_uint

const cl_sampler_properties = cl_properties

const cl_kernel_exec_info = cl_uint

const cl_device_atomic_capabilities = cl_bitfield

const cl_device_device_enqueue_capabilities = cl_bitfield

const cl_khronos_vendor_id = cl_uint

const cl_mem_properties = cl_properties

const cl_version = cl_uint

struct _cl_image_format
    image_channel_order::cl_channel_order
    image_channel_data_type::cl_channel_type
end

const cl_image_format = _cl_image_format

struct _cl_image_desc
    data::NTuple{72,UInt8}
end

function Base.getproperty(x::Ptr{_cl_image_desc}, f::Symbol)
    f === :image_type && return Ptr{cl_mem_object_type}(x + 0)
    f === :image_width && return Ptr{Csize_t}(x + 8)
    f === :image_height && return Ptr{Csize_t}(x + 16)
    f === :image_depth && return Ptr{Csize_t}(x + 24)
    f === :image_array_size && return Ptr{Csize_t}(x + 32)
    f === :image_row_pitch && return Ptr{Csize_t}(x + 40)
    f === :image_slice_pitch && return Ptr{Csize_t}(x + 48)
    f === :num_mip_levels && return Ptr{cl_uint}(x + 56)
    f === :num_samples && return Ptr{cl_uint}(x + 60)
    f === :buffer && return Ptr{cl_mem}(x + 64)
    f === :mem_object && return Ptr{cl_mem}(x + 64)
    return getfield(x, f)
end

function Base.getproperty(x::_cl_image_desc, f::Symbol)
    r = Ref{_cl_image_desc}(x)
    ptr = Base.unsafe_convert(Ptr{_cl_image_desc}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{_cl_image_desc}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

const cl_image_desc = _cl_image_desc

struct _cl_buffer_region
    origin::Csize_t
    size::Csize_t
end

const cl_buffer_region = _cl_buffer_region

struct _cl_name_version
    version::cl_version
    name::NTuple{64,Cchar}
end

const cl_name_version = _cl_name_version

@checked function clGetPlatformIDs(num_entries, platforms, num_platforms)
    @ccall libopencl.clGetPlatformIDs(num_entries::cl_uint, platforms::Ptr{cl_platform_id},
                                      num_platforms::Ptr{cl_uint})::cl_int
end

@checked function clGetPlatformInfo(platform, param_name, param_value_size, param_value,
                                    param_value_size_ret)
    @ccall libopencl.clGetPlatformInfo(platform::cl_platform_id,
                                       param_name::cl_platform_info,
                                       param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                       param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices)
    @ccall libopencl.clGetDeviceIDs(platform::cl_platform_id, device_type::cl_device_type,
                                    num_entries::cl_uint, devices::Ptr{cl_device_id},
                                    num_devices::Ptr{cl_uint})::cl_int
end

@checked function clGetDeviceInfo(device, param_name, param_value_size, param_value,
                                  param_value_size_ret)
    @ccall libopencl.clGetDeviceInfo(device::cl_device_id, param_name::cl_device_info,
                                     param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                     param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clCreateSubDevices(in_device, properties, num_devices, out_devices,
                                     num_devices_ret)
    @ccall libopencl.clCreateSubDevices(in_device::cl_device_id,
                                        properties::Ptr{cl_device_partition_property},
                                        num_devices::cl_uint,
                                        out_devices::Ptr{cl_device_id},
                                        num_devices_ret::Ptr{cl_uint})::cl_int
end

@checked function clRetainDevice(device)
    @ccall libopencl.clRetainDevice(device::cl_device_id)::cl_int
end

@checked function clReleaseDevice(device)
    @ccall libopencl.clReleaseDevice(device::cl_device_id)::cl_int
end

@checked function clSetDefaultDeviceCommandQueue(context, device, command_queue)
    @ccall libopencl.clSetDefaultDeviceCommandQueue(context::cl_context,
                                                    device::cl_device_id,
                                                    command_queue::cl_command_queue)::cl_int
end

@checked function clGetDeviceAndHostTimer(device, device_timestamp, host_timestamp)
    @ccall libopencl.clGetDeviceAndHostTimer(device::cl_device_id,
                                             device_timestamp::Ptr{cl_ulong},
                                             host_timestamp::Ptr{cl_ulong})::cl_int
end

@checked function clGetHostTimer(device, host_timestamp)
    @ccall libopencl.clGetHostTimer(device::cl_device_id,
                                    host_timestamp::Ptr{cl_ulong})::cl_int
end

function clCreateContext(properties, num_devices, devices, pfn_notify, user_data,
                         errcode_ret)
    @ccall libopencl.clCreateContext(properties::Ptr{cl_context_properties},
                                     num_devices::cl_uint, devices::Ptr{cl_device_id},
                                     pfn_notify::Ptr{Cvoid}, user_data::Ptr{Cvoid},
                                     errcode_ret::Ptr{cl_int})::cl_context
end

function clCreateContextFromType(properties, device_type, pfn_notify, user_data,
                                 errcode_ret)
    @ccall libopencl.clCreateContextFromType(properties::Ptr{cl_context_properties},
                                             device_type::cl_device_type,
                                             pfn_notify::Ptr{Cvoid}, user_data::Ptr{Cvoid},
                                             errcode_ret::Ptr{cl_int})::cl_context
end

@checked function clRetainContext(context)
    @ccall libopencl.clRetainContext(context::cl_context)::cl_int
end

@checked function clReleaseContext(context)
    @ccall libopencl.clReleaseContext(context::cl_context)::cl_int
end

@checked function clGetContextInfo(context, param_name, param_value_size, param_value,
                                   param_value_size_ret)
    @ccall libopencl.clGetContextInfo(context::cl_context, param_name::cl_context_info,
                                      param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                      param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clSetContextDestructorCallback(context, pfn_notify, user_data)
    @ccall libopencl.clSetContextDestructorCallback(context::cl_context,
                                                    pfn_notify::Ptr{Cvoid},
                                                    user_data::Ptr{Cvoid})::cl_int
end

function clCreateCommandQueueWithProperties(context, device, properties, errcode_ret)
    @ccall libopencl.clCreateCommandQueueWithProperties(context::cl_context,
                                                        device::cl_device_id,
                                                        properties::Ptr{cl_queue_properties},
                                                        errcode_ret::Ptr{cl_int})::cl_command_queue
end

@checked function clRetainCommandQueue(command_queue)
    @ccall libopencl.clRetainCommandQueue(command_queue::cl_command_queue)::cl_int
end

@checked function clReleaseCommandQueue(command_queue)
    @ccall libopencl.clReleaseCommandQueue(command_queue::cl_command_queue)::cl_int
end

@checked function clGetCommandQueueInfo(command_queue, param_name, param_value_size,
                                        param_value, param_value_size_ret)
    @ccall libopencl.clGetCommandQueueInfo(command_queue::cl_command_queue,
                                           param_name::cl_command_queue_info,
                                           param_value_size::Csize_t,
                                           param_value::Ptr{Cvoid},
                                           param_value_size_ret::Ptr{Csize_t})::cl_int
end

function clCreateBuffer(context, flags, size, host_ptr, errcode_ret)
    @ccall libopencl.clCreateBuffer(context::cl_context, flags::cl_mem_flags, size::Csize_t,
                                    host_ptr::Ptr{Cvoid}, errcode_ret::Ptr{cl_int})::cl_mem
end

function clCreateSubBuffer(buffer, flags, buffer_create_type, buffer_create_info,
                           errcode_ret)
    @ccall libopencl.clCreateSubBuffer(buffer::cl_mem, flags::cl_mem_flags,
                                       buffer_create_type::cl_buffer_create_type,
                                       buffer_create_info::Ptr{Cvoid},
                                       errcode_ret::Ptr{cl_int})::cl_mem
end

function clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret)
    @ccall libopencl.clCreateImage(context::cl_context, flags::cl_mem_flags,
                                   image_format::Ptr{cl_image_format},
                                   image_desc::Ptr{cl_image_desc}, host_ptr::Ptr{Cvoid},
                                   errcode_ret::Ptr{cl_int})::cl_mem
end

function clCreatePipe(context, flags, pipe_packet_size, pipe_max_packets, properties,
                      errcode_ret)
    @ccall libopencl.clCreatePipe(context::cl_context, flags::cl_mem_flags,
                                  pipe_packet_size::cl_uint, pipe_max_packets::cl_uint,
                                  properties::Ptr{cl_pipe_properties},
                                  errcode_ret::Ptr{cl_int})::cl_mem
end

function clCreateBufferWithProperties(context, properties, flags, size, host_ptr,
                                      errcode_ret)
    @ccall libopencl.clCreateBufferWithProperties(context::cl_context,
                                                  properties::Ptr{cl_mem_properties},
                                                  flags::cl_mem_flags, size::Csize_t,
                                                  host_ptr::Ptr{Cvoid},
                                                  errcode_ret::Ptr{cl_int})::cl_mem
end

function clCreateImageWithProperties(context, properties, flags, image_format, image_desc,
                                     host_ptr, errcode_ret)
    @ccall libopencl.clCreateImageWithProperties(context::cl_context,
                                                 properties::Ptr{cl_mem_properties},
                                                 flags::cl_mem_flags,
                                                 image_format::Ptr{cl_image_format},
                                                 image_desc::Ptr{cl_image_desc},
                                                 host_ptr::Ptr{Cvoid},
                                                 errcode_ret::Ptr{cl_int})::cl_mem
end

@checked function clRetainMemObject(memobj)
    @ccall libopencl.clRetainMemObject(memobj::cl_mem)::cl_int
end

@checked function clReleaseMemObject(memobj)
    @ccall libopencl.clReleaseMemObject(memobj::cl_mem)::cl_int
end

@checked function clGetSupportedImageFormats(context, flags, image_type, num_entries,
                                             image_formats, num_image_formats)
    @ccall libopencl.clGetSupportedImageFormats(context::cl_context, flags::cl_mem_flags,
                                                image_type::cl_mem_object_type,
                                                num_entries::cl_uint,
                                                image_formats::Ptr{cl_image_format},
                                                num_image_formats::Ptr{cl_uint})::cl_int
end

@checked function clGetMemObjectInfo(memobj, param_name, param_value_size, param_value,
                                     param_value_size_ret)
    @ccall libopencl.clGetMemObjectInfo(memobj::cl_mem, param_name::cl_mem_info,
                                        param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                        param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clGetImageInfo(image, param_name, param_value_size, param_value,
                                 param_value_size_ret)
    @ccall libopencl.clGetImageInfo(image::cl_mem, param_name::cl_image_info,
                                    param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                    param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clGetPipeInfo(pipe, param_name, param_value_size, param_value,
                                param_value_size_ret)
    @ccall libopencl.clGetPipeInfo(pipe::cl_mem, param_name::cl_pipe_info,
                                   param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                   param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clSetMemObjectDestructorCallback(memobj, pfn_notify, user_data)
    @ccall libopencl.clSetMemObjectDestructorCallback(memobj::cl_mem,
                                                      pfn_notify::Ptr{Cvoid},
                                                      user_data::Ptr{Cvoid})::cl_int
end

function clSVMAlloc(context, flags, size, alignment)
    @ccall libopencl.clSVMAlloc(context::cl_context, flags::cl_svm_mem_flags, size::Csize_t,
                                alignment::cl_uint)::Ptr{Cvoid}
end

function clSVMFree(context, svm_pointer)
    @ccall libopencl.clSVMFree(context::cl_context, svm_pointer::Ptr{Cvoid})::Cvoid
end

function clCreateSamplerWithProperties(context, sampler_properties, errcode_ret)
    @ccall libopencl.clCreateSamplerWithProperties(context::cl_context,
                                                   sampler_properties::Ptr{cl_sampler_properties},
                                                   errcode_ret::Ptr{cl_int})::cl_sampler
end

@checked function clRetainSampler(sampler)
    @ccall libopencl.clRetainSampler(sampler::cl_sampler)::cl_int
end

@checked function clReleaseSampler(sampler)
    @ccall libopencl.clReleaseSampler(sampler::cl_sampler)::cl_int
end

@checked function clGetSamplerInfo(sampler, param_name, param_value_size, param_value,
                                   param_value_size_ret)
    @ccall libopencl.clGetSamplerInfo(sampler::cl_sampler, param_name::cl_sampler_info,
                                      param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                      param_value_size_ret::Ptr{Csize_t})::cl_int
end

function clCreateProgramWithSource(context, count, strings, lengths, errcode_ret)
    @ccall libopencl.clCreateProgramWithSource(context::cl_context, count::cl_uint,
                                               strings::Ptr{Ptr{Cchar}},
                                               lengths::Ptr{Csize_t},
                                               errcode_ret::Ptr{cl_int})::cl_program
end

function clCreateProgramWithBinary(context, num_devices, device_list, lengths, binaries,
                                   binary_status, errcode_ret)
    @ccall libopencl.clCreateProgramWithBinary(context::cl_context, num_devices::cl_uint,
                                               device_list::Ptr{cl_device_id},
                                               lengths::Ptr{Csize_t},
                                               binaries::Ptr{Ptr{Cuchar}},
                                               binary_status::Ptr{cl_int},
                                               errcode_ret::Ptr{cl_int})::cl_program
end

function clCreateProgramWithBuiltInKernels(context, num_devices, device_list, kernel_names,
                                           errcode_ret)
    @ccall libopencl.clCreateProgramWithBuiltInKernels(context::cl_context,
                                                       num_devices::cl_uint,
                                                       device_list::Ptr{cl_device_id},
                                                       kernel_names::Ptr{Cchar},
                                                       errcode_ret::Ptr{cl_int})::cl_program
end

function clCreateProgramWithIL(context, il, length, errcode_ret)
    @ccall libopencl.clCreateProgramWithIL(context::cl_context, il::Ptr{Cvoid},
                                           length::Csize_t,
                                           errcode_ret::Ptr{cl_int})::cl_program
end

@checked function clRetainProgram(program)
    @ccall libopencl.clRetainProgram(program::cl_program)::cl_int
end

@checked function clReleaseProgram(program)
    @ccall libopencl.clReleaseProgram(program::cl_program)::cl_int
end

@checked function clBuildProgram(program, num_devices, device_list, options, pfn_notify,
                                 user_data)
    @ccall libopencl.clBuildProgram(program::cl_program, num_devices::cl_uint,
                                    device_list::Ptr{cl_device_id}, options::Ptr{Cchar},
                                    pfn_notify::Ptr{Cvoid}, user_data::Ptr{Cvoid})::cl_int
end

@checked function clCompileProgram(program, num_devices, device_list, options,
                                   num_input_headers, input_headers, header_include_names,
                                   pfn_notify, user_data)
    @ccall libopencl.clCompileProgram(program::cl_program, num_devices::cl_uint,
                                      device_list::Ptr{cl_device_id}, options::Ptr{Cchar},
                                      num_input_headers::cl_uint,
                                      input_headers::Ptr{cl_program},
                                      header_include_names::Ptr{Ptr{Cchar}},
                                      pfn_notify::Ptr{Cvoid}, user_data::Ptr{Cvoid})::cl_int
end

function clLinkProgram(context, num_devices, device_list, options, num_input_programs,
                       input_programs, pfn_notify, user_data, errcode_ret)
    @ccall libopencl.clLinkProgram(context::cl_context, num_devices::cl_uint,
                                   device_list::Ptr{cl_device_id}, options::Ptr{Cchar},
                                   num_input_programs::cl_uint,
                                   input_programs::Ptr{cl_program}, pfn_notify::Ptr{Cvoid},
                                   user_data::Ptr{Cvoid},
                                   errcode_ret::Ptr{cl_int})::cl_program
end

@checked function clSetProgramReleaseCallback(program, pfn_notify, user_data)
    @ccall libopencl.clSetProgramReleaseCallback(program::cl_program,
                                                 pfn_notify::Ptr{Cvoid},
                                                 user_data::Ptr{Cvoid})::cl_int
end

@checked function clSetProgramSpecializationConstant(program, spec_id, spec_size,
                                                     spec_value)
    @ccall libopencl.clSetProgramSpecializationConstant(program::cl_program,
                                                        spec_id::cl_uint,
                                                        spec_size::Csize_t,
                                                        spec_value::Ptr{Cvoid})::cl_int
end

@checked function clUnloadPlatformCompiler(platform)
    @ccall libopencl.clUnloadPlatformCompiler(platform::cl_platform_id)::cl_int
end

@checked function clGetProgramInfo(program, param_name, param_value_size, param_value,
                                   param_value_size_ret)
    @ccall libopencl.clGetProgramInfo(program::cl_program, param_name::cl_program_info,
                                      param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                      param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clGetProgramBuildInfo(program, device, param_name, param_value_size,
                                        param_value, param_value_size_ret)
    @ccall libopencl.clGetProgramBuildInfo(program::cl_program, device::cl_device_id,
                                           param_name::cl_program_build_info,
                                           param_value_size::Csize_t,
                                           param_value::Ptr{Cvoid},
                                           param_value_size_ret::Ptr{Csize_t})::cl_int
end

function clCreateKernel(program, kernel_name, errcode_ret)
    @ccall libopencl.clCreateKernel(program::cl_program, kernel_name::Ptr{Cchar},
                                    errcode_ret::Ptr{cl_int})::cl_kernel
end

@checked function clCreateKernelsInProgram(program, num_kernels, kernels, num_kernels_ret)
    @ccall libopencl.clCreateKernelsInProgram(program::cl_program, num_kernels::cl_uint,
                                              kernels::Ptr{cl_kernel},
                                              num_kernels_ret::Ptr{cl_uint})::cl_int
end

function clCloneKernel(source_kernel, errcode_ret)
    @ccall libopencl.clCloneKernel(source_kernel::cl_kernel,
                                   errcode_ret::Ptr{cl_int})::cl_kernel
end

@checked function clRetainKernel(kernel)
    @ccall libopencl.clRetainKernel(kernel::cl_kernel)::cl_int
end

@checked function clReleaseKernel(kernel)
    @ccall libopencl.clReleaseKernel(kernel::cl_kernel)::cl_int
end

@checked function clSetKernelArg(kernel, arg_index, arg_size, arg_value)
    @ccall libopencl.clSetKernelArg(kernel::cl_kernel, arg_index::cl_uint,
                                    arg_size::Csize_t, arg_value::Ptr{Cvoid})::cl_int
end

@checked function clSetKernelArgSVMPointer(kernel, arg_index, arg_value)
    @ccall libopencl.clSetKernelArgSVMPointer(kernel::cl_kernel, arg_index::cl_uint,
                                              arg_value::Ptr{Cvoid})::cl_int
end

@checked function clSetKernelExecInfo(kernel, param_name, param_value_size, param_value)
    @ccall libopencl.clSetKernelExecInfo(kernel::cl_kernel, param_name::cl_kernel_exec_info,
                                         param_value_size::Csize_t,
                                         param_value::Ptr{Cvoid})::cl_int
end

@checked function clGetKernelInfo(kernel, param_name, param_value_size, param_value,
                                  param_value_size_ret)
    @ccall libopencl.clGetKernelInfo(kernel::cl_kernel, param_name::cl_kernel_info,
                                     param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                     param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clGetKernelArgInfo(kernel, arg_indx, param_name, param_value_size,
                                     param_value, param_value_size_ret)
    @ccall libopencl.clGetKernelArgInfo(kernel::cl_kernel, arg_indx::cl_uint,
                                        param_name::cl_kernel_arg_info,
                                        param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                        param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clGetKernelWorkGroupInfo(kernel, device, param_name, param_value_size,
                                           param_value, param_value_size_ret)
    @ccall libopencl.clGetKernelWorkGroupInfo(kernel::cl_kernel, device::cl_device_id,
                                              param_name::cl_kernel_work_group_info,
                                              param_value_size::Csize_t,
                                              param_value::Ptr{Cvoid},
                                              param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clGetKernelSubGroupInfo(kernel, device, param_name, input_value_size,
                                          input_value, param_value_size, param_value,
                                          param_value_size_ret)
    @ccall libopencl.clGetKernelSubGroupInfo(kernel::cl_kernel, device::cl_device_id,
                                             param_name::cl_kernel_sub_group_info,
                                             input_value_size::Csize_t,
                                             input_value::Ptr{Cvoid},
                                             param_value_size::Csize_t,
                                             param_value::Ptr{Cvoid},
                                             param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clWaitForEvents(num_events, event_list)
    @ccall libopencl.clWaitForEvents(num_events::cl_uint, event_list::Ptr{cl_event})::cl_int
end

@checked function clGetEventInfo(event, param_name, param_value_size, param_value,
                                 param_value_size_ret)
    @ccall libopencl.clGetEventInfo(event::cl_event, param_name::cl_event_info,
                                    param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                    param_value_size_ret::Ptr{Csize_t})::cl_int
end

function clCreateUserEvent(context, errcode_ret)
    @ccall libopencl.clCreateUserEvent(context::cl_context,
                                       errcode_ret::Ptr{cl_int})::cl_event
end

@checked function clRetainEvent(event)
    @ccall libopencl.clRetainEvent(event::cl_event)::cl_int
end

@checked function clReleaseEvent(event)
    @ccall libopencl.clReleaseEvent(event::cl_event)::cl_int
end

@checked function clSetUserEventStatus(event, execution_status)
    @ccall libopencl.clSetUserEventStatus(event::cl_event, execution_status::cl_int)::cl_int
end

@checked function clSetEventCallback(event, command_exec_callback_type, pfn_notify,
                                     user_data)
    @ccall libopencl.clSetEventCallback(event::cl_event, command_exec_callback_type::cl_int,
                                        pfn_notify::Ptr{Cvoid},
                                        user_data::Ptr{Cvoid})::cl_int
end

@checked function clGetEventProfilingInfo(event, param_name, param_value_size, param_value,
                                          param_value_size_ret)
    @ccall libopencl.clGetEventProfilingInfo(event::cl_event, param_name::cl_profiling_info,
                                             param_value_size::Csize_t,
                                             param_value::Ptr{Cvoid},
                                             param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clFlush(command_queue)
    @ccall libopencl.clFlush(command_queue::cl_command_queue)::cl_int
end

@checked function clFinish(command_queue)
    @ccall libopencl.clFinish(command_queue::cl_command_queue)::cl_int
end

@checked function clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size,
                                      ptr, num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueReadBuffer(command_queue::cl_command_queue, buffer::cl_mem,
                                         blocking_read::cl_bool, offset::Csize_t,
                                         size::Csize_t, ptr::Ptr{Cvoid},
                                         num_events_in_wait_list::cl_uint,
                                         event_wait_list::Ptr{cl_event},
                                         event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueReadBufferRect(command_queue, buffer, blocking_read,
                                          buffer_origin, host_origin, region,
                                          buffer_row_pitch, buffer_slice_pitch,
                                          host_row_pitch, host_slice_pitch, ptr,
                                          num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueReadBufferRect(command_queue::cl_command_queue,
                                             buffer::cl_mem, blocking_read::cl_bool,
                                             buffer_origin::Ptr{Csize_t},
                                             host_origin::Ptr{Csize_t},
                                             region::Ptr{Csize_t},
                                             buffer_row_pitch::Csize_t,
                                             buffer_slice_pitch::Csize_t,
                                             host_row_pitch::Csize_t,
                                             host_slice_pitch::Csize_t, ptr::Ptr{Cvoid},
                                             num_events_in_wait_list::cl_uint,
                                             event_wait_list::Ptr{cl_event},
                                             event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, size,
                                       ptr, num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueWriteBuffer(command_queue::cl_command_queue, buffer::cl_mem,
                                          blocking_write::cl_bool, offset::Csize_t,
                                          size::Csize_t, ptr::Ptr{Cvoid},
                                          num_events_in_wait_list::cl_uint,
                                          event_wait_list::Ptr{cl_event},
                                          event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueWriteBufferRect(command_queue, buffer, blocking_write,
                                           buffer_origin, host_origin, region,
                                           buffer_row_pitch, buffer_slice_pitch,
                                           host_row_pitch, host_slice_pitch, ptr,
                                           num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueWriteBufferRect(command_queue::cl_command_queue,
                                              buffer::cl_mem, blocking_write::cl_bool,
                                              buffer_origin::Ptr{Csize_t},
                                              host_origin::Ptr{Csize_t},
                                              region::Ptr{Csize_t},
                                              buffer_row_pitch::Csize_t,
                                              buffer_slice_pitch::Csize_t,
                                              host_row_pitch::Csize_t,
                                              host_slice_pitch::Csize_t, ptr::Ptr{Cvoid},
                                              num_events_in_wait_list::cl_uint,
                                              event_wait_list::Ptr{cl_event},
                                              event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueFillBuffer(command_queue, buffer, pattern, pattern_size, offset,
                                      size, num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueFillBuffer(command_queue::cl_command_queue, buffer::cl_mem,
                                         pattern::Ptr{Cvoid}, pattern_size::Csize_t,
                                         offset::Csize_t, size::Csize_t,
                                         num_events_in_wait_list::cl_uint,
                                         event_wait_list::Ptr{cl_event},
                                         event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueCopyBuffer(command_queue, src_buffer, dst_buffer, src_offset,
                                      dst_offset, size, num_events_in_wait_list,
                                      event_wait_list, event)
    @ccall libopencl.clEnqueueCopyBuffer(command_queue::cl_command_queue,
                                         src_buffer::cl_mem, dst_buffer::cl_mem,
                                         src_offset::Csize_t, dst_offset::Csize_t,
                                         size::Csize_t, num_events_in_wait_list::cl_uint,
                                         event_wait_list::Ptr{cl_event},
                                         event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueCopyBufferRect(command_queue, src_buffer, dst_buffer, src_origin,
                                          dst_origin, region, src_row_pitch,
                                          src_slice_pitch, dst_row_pitch, dst_slice_pitch,
                                          num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueCopyBufferRect(command_queue::cl_command_queue,
                                             src_buffer::cl_mem, dst_buffer::cl_mem,
                                             src_origin::Ptr{Csize_t},
                                             dst_origin::Ptr{Csize_t}, region::Ptr{Csize_t},
                                             src_row_pitch::Csize_t,
                                             src_slice_pitch::Csize_t,
                                             dst_row_pitch::Csize_t,
                                             dst_slice_pitch::Csize_t,
                                             num_events_in_wait_list::cl_uint,
                                             event_wait_list::Ptr{cl_event},
                                             event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueReadImage(command_queue, image, blocking_read, origin, region,
                                     row_pitch, slice_pitch, ptr, num_events_in_wait_list,
                                     event_wait_list, event)
    @ccall libopencl.clEnqueueReadImage(command_queue::cl_command_queue, image::cl_mem,
                                        blocking_read::cl_bool, origin::Ptr{Csize_t},
                                        region::Ptr{Csize_t}, row_pitch::Csize_t,
                                        slice_pitch::Csize_t, ptr::Ptr{Cvoid},
                                        num_events_in_wait_list::cl_uint,
                                        event_wait_list::Ptr{cl_event},
                                        event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueWriteImage(command_queue, image, blocking_write, origin, region,
                                      input_row_pitch, input_slice_pitch, ptr,
                                      num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueWriteImage(command_queue::cl_command_queue, image::cl_mem,
                                         blocking_write::cl_bool, origin::Ptr{Csize_t},
                                         region::Ptr{Csize_t}, input_row_pitch::Csize_t,
                                         input_slice_pitch::Csize_t, ptr::Ptr{Cvoid},
                                         num_events_in_wait_list::cl_uint,
                                         event_wait_list::Ptr{cl_event},
                                         event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueFillImage(command_queue, image, fill_color, origin, region,
                                     num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueFillImage(command_queue::cl_command_queue, image::cl_mem,
                                        fill_color::Ptr{Cvoid}, origin::Ptr{Csize_t},
                                        region::Ptr{Csize_t},
                                        num_events_in_wait_list::cl_uint,
                                        event_wait_list::Ptr{cl_event},
                                        event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueCopyImage(command_queue, src_image, dst_image, src_origin,
                                     dst_origin, region, num_events_in_wait_list,
                                     event_wait_list, event)
    @ccall libopencl.clEnqueueCopyImage(command_queue::cl_command_queue, src_image::cl_mem,
                                        dst_image::cl_mem, src_origin::Ptr{Csize_t},
                                        dst_origin::Ptr{Csize_t}, region::Ptr{Csize_t},
                                        num_events_in_wait_list::cl_uint,
                                        event_wait_list::Ptr{cl_event},
                                        event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueCopyImageToBuffer(command_queue, src_image, dst_buffer,
                                             src_origin, region, dst_offset,
                                             num_events_in_wait_list, event_wait_list,
                                             event)
    @ccall libopencl.clEnqueueCopyImageToBuffer(command_queue::cl_command_queue,
                                                src_image::cl_mem, dst_buffer::cl_mem,
                                                src_origin::Ptr{Csize_t},
                                                region::Ptr{Csize_t}, dst_offset::Csize_t,
                                                num_events_in_wait_list::cl_uint,
                                                event_wait_list::Ptr{cl_event},
                                                event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueCopyBufferToImage(command_queue, src_buffer, dst_image,
                                             src_offset, dst_origin, region,
                                             num_events_in_wait_list, event_wait_list,
                                             event)
    @ccall libopencl.clEnqueueCopyBufferToImage(command_queue::cl_command_queue,
                                                src_buffer::cl_mem, dst_image::cl_mem,
                                                src_offset::Csize_t,
                                                dst_origin::Ptr{Csize_t},
                                                region::Ptr{Csize_t},
                                                num_events_in_wait_list::cl_uint,
                                                event_wait_list::Ptr{cl_event},
                                                event::Ptr{cl_event})::cl_int
end

function clEnqueueMapBuffer(command_queue, buffer, blocking_map, map_flags, offset, size,
                            num_events_in_wait_list, event_wait_list, event, errcode_ret)
    @ccall libopencl.clEnqueueMapBuffer(command_queue::cl_command_queue, buffer::cl_mem,
                                        blocking_map::cl_bool, map_flags::cl_map_flags,
                                        offset::Csize_t, size::Csize_t,
                                        num_events_in_wait_list::cl_uint,
                                        event_wait_list::Ptr{cl_event},
                                        event::Ptr{cl_event},
                                        errcode_ret::Ptr{cl_int})::Ptr{Cvoid}
end

function clEnqueueMapImage(command_queue, image, blocking_map, map_flags, origin, region,
                           image_row_pitch, image_slice_pitch, num_events_in_wait_list,
                           event_wait_list, event, errcode_ret)
    @ccall libopencl.clEnqueueMapImage(command_queue::cl_command_queue, image::cl_mem,
                                       blocking_map::cl_bool, map_flags::cl_map_flags,
                                       origin::Ptr{Csize_t}, region::Ptr{Csize_t},
                                       image_row_pitch::Ptr{Csize_t},
                                       image_slice_pitch::Ptr{Csize_t},
                                       num_events_in_wait_list::cl_uint,
                                       event_wait_list::Ptr{cl_event}, event::Ptr{cl_event},
                                       errcode_ret::Ptr{cl_int})::Ptr{Cvoid}
end

@checked function clEnqueueUnmapMemObject(command_queue, memobj, mapped_ptr,
                                          num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueUnmapMemObject(command_queue::cl_command_queue,
                                             memobj::cl_mem, mapped_ptr::Ptr{Cvoid},
                                             num_events_in_wait_list::cl_uint,
                                             event_wait_list::Ptr{cl_event},
                                             event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueMigrateMemObjects(command_queue, num_mem_objects, mem_objects,
                                             flags, num_events_in_wait_list,
                                             event_wait_list, event)
    @ccall libopencl.clEnqueueMigrateMemObjects(command_queue::cl_command_queue,
                                                num_mem_objects::cl_uint,
                                                mem_objects::Ptr{cl_mem},
                                                flags::cl_mem_migration_flags,
                                                num_events_in_wait_list::cl_uint,
                                                event_wait_list::Ptr{cl_event},
                                                event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                         global_work_offset, global_work_size,
                                         local_work_size, num_events_in_wait_list,
                                         event_wait_list, event)
    @ccall libopencl.clEnqueueNDRangeKernel(command_queue::cl_command_queue,
                                            kernel::cl_kernel, work_dim::cl_uint,
                                            global_work_offset::Ptr{Csize_t},
                                            global_work_size::Ptr{Csize_t},
                                            local_work_size::Ptr{Csize_t},
                                            num_events_in_wait_list::cl_uint,
                                            event_wait_list::Ptr{cl_event},
                                            event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueNativeKernel(command_queue, user_func, args, cb_args,
                                        num_mem_objects, mem_list, args_mem_loc,
                                        num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueNativeKernel(command_queue::cl_command_queue,
                                           user_func::Ptr{Cvoid}, args::Ptr{Cvoid},
                                           cb_args::Csize_t, num_mem_objects::cl_uint,
                                           mem_list::Ptr{cl_mem},
                                           args_mem_loc::Ptr{Ptr{Cvoid}},
                                           num_events_in_wait_list::cl_uint,
                                           event_wait_list::Ptr{cl_event},
                                           event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueMarkerWithWaitList(command_queue, num_events_in_wait_list,
                                              event_wait_list, event)
    @ccall libopencl.clEnqueueMarkerWithWaitList(command_queue::cl_command_queue,
                                                 num_events_in_wait_list::cl_uint,
                                                 event_wait_list::Ptr{cl_event},
                                                 event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueBarrierWithWaitList(command_queue, num_events_in_wait_list,
                                               event_wait_list, event)
    @ccall libopencl.clEnqueueBarrierWithWaitList(command_queue::cl_command_queue,
                                                  num_events_in_wait_list::cl_uint,
                                                  event_wait_list::Ptr{cl_event},
                                                  event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMFree(command_queue, num_svm_pointers, svm_pointers,
                                   pfn_free_func, user_data, num_events_in_wait_list,
                                   event_wait_list, event)
    @ccall libopencl.clEnqueueSVMFree(command_queue::cl_command_queue,
                                      num_svm_pointers::cl_uint,
                                      svm_pointers::Ptr{Ptr{Cvoid}},
                                      pfn_free_func::Ptr{Cvoid}, user_data::Ptr{Cvoid},
                                      num_events_in_wait_list::cl_uint,
                                      event_wait_list::Ptr{cl_event},
                                      event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMMemcpy(command_queue, blocking_copy, dst_ptr, src_ptr, size,
                                     num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueSVMMemcpy(command_queue::cl_command_queue,
                                        blocking_copy::cl_bool, dst_ptr::Ptr{Cvoid},
                                        src_ptr::Ptr{Cvoid}, size::Csize_t,
                                        num_events_in_wait_list::cl_uint,
                                        event_wait_list::Ptr{cl_event},
                                        event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMMemFill(command_queue, svm_ptr, pattern, pattern_size, size,
                                      num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueSVMMemFill(command_queue::cl_command_queue,
                                         svm_ptr::Ptr{Cvoid}, pattern::Ptr{Cvoid},
                                         pattern_size::Csize_t, size::Csize_t,
                                         num_events_in_wait_list::cl_uint,
                                         event_wait_list::Ptr{cl_event},
                                         event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMMap(command_queue, blocking_map, flags, svm_ptr, size,
                                  num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueSVMMap(command_queue::cl_command_queue, blocking_map::cl_bool,
                                     flags::cl_map_flags, svm_ptr::Ptr{Cvoid},
                                     size::Csize_t, num_events_in_wait_list::cl_uint,
                                     event_wait_list::Ptr{cl_event},
                                     event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMUnmap(command_queue, svm_ptr, num_events_in_wait_list,
                                    event_wait_list, event)
    @ccall libopencl.clEnqueueSVMUnmap(command_queue::cl_command_queue, svm_ptr::Ptr{Cvoid},
                                       num_events_in_wait_list::cl_uint,
                                       event_wait_list::Ptr{cl_event},
                                       event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMMigrateMem(command_queue, num_svm_pointers, svm_pointers,
                                         sizes, flags, num_events_in_wait_list,
                                         event_wait_list, event)
    @ccall libopencl.clEnqueueSVMMigrateMem(command_queue::cl_command_queue,
                                            num_svm_pointers::cl_uint,
                                            svm_pointers::Ptr{Ptr{Cvoid}},
                                            sizes::Ptr{Csize_t},
                                            flags::cl_mem_migration_flags,
                                            num_events_in_wait_list::cl_uint,
                                            event_wait_list::Ptr{cl_event},
                                            event::Ptr{cl_event})::cl_int
end

function clGetExtensionFunctionAddressForPlatform(platform, func_name)
    @ccall libopencl.clGetExtensionFunctionAddressForPlatform(platform::cl_platform_id,
                                                              func_name::Ptr{Cchar})::Ptr{Cvoid}
end

function clCreateImage2D(context, flags, image_format, image_width, image_height,
                         image_row_pitch, host_ptr, errcode_ret)
    @ccall libopencl.clCreateImage2D(context::cl_context, flags::cl_mem_flags,
                                     image_format::Ptr{cl_image_format},
                                     image_width::Csize_t, image_height::Csize_t,
                                     image_row_pitch::Csize_t, host_ptr::Ptr{Cvoid},
                                     errcode_ret::Ptr{cl_int})::cl_mem
end

function clCreateImage3D(context, flags, image_format, image_width, image_height,
                         image_depth, image_row_pitch, image_slice_pitch, host_ptr,
                         errcode_ret)
    @ccall libopencl.clCreateImage3D(context::cl_context, flags::cl_mem_flags,
                                     image_format::Ptr{cl_image_format},
                                     image_width::Csize_t, image_height::Csize_t,
                                     image_depth::Csize_t, image_row_pitch::Csize_t,
                                     image_slice_pitch::Csize_t, host_ptr::Ptr{Cvoid},
                                     errcode_ret::Ptr{cl_int})::cl_mem
end

@checked function clEnqueueMarker(command_queue, event)
    @ccall libopencl.clEnqueueMarker(command_queue::cl_command_queue,
                                     event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueWaitForEvents(command_queue, num_events, event_list)
    @ccall libopencl.clEnqueueWaitForEvents(command_queue::cl_command_queue,
                                            num_events::cl_uint,
                                            event_list::Ptr{cl_event})::cl_int
end

@checked function clEnqueueBarrier(command_queue)
    @ccall libopencl.clEnqueueBarrier(command_queue::cl_command_queue)::cl_int
end

@checked function clUnloadCompiler()
    @ccall libopencl.clUnloadCompiler()::cl_int
end

function clGetExtensionFunctionAddress(func_name)
    @ccall libopencl.clGetExtensionFunctionAddress(func_name::Ptr{Cchar})::Ptr{Cvoid}
end

function clCreateCommandQueue(context, device, properties, errcode_ret)
    @ccall libopencl.clCreateCommandQueue(context::cl_context, device::cl_device_id,
                                          properties::cl_command_queue_properties,
                                          errcode_ret::Ptr{cl_int})::cl_command_queue
end

function clCreateSampler(context, normalized_coords, addressing_mode, filter_mode,
                         errcode_ret)
    @ccall libopencl.clCreateSampler(context::cl_context, normalized_coords::cl_bool,
                                     addressing_mode::cl_addressing_mode,
                                     filter_mode::cl_filter_mode,
                                     errcode_ret::Ptr{cl_int})::cl_sampler
end

@checked function clEnqueueTask(command_queue, kernel, num_events_in_wait_list,
                                event_wait_list, event)
    @ccall libopencl.clEnqueueTask(command_queue::cl_command_queue, kernel::cl_kernel,
                                   num_events_in_wait_list::cl_uint,
                                   event_wait_list::Ptr{cl_event},
                                   event::Ptr{cl_event})::cl_int
end

const cl_gl_context_info = cl_uint

const cl_gl_object_type = cl_uint

const cl_gl_texture_info = cl_uint

const cl_gl_platform_info = cl_uint

# typedef cl_int CL_API_CALL clGetGLContextInfoKHR_t ( const cl_context_properties * properties , cl_gl_context_info param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetGLContextInfoKHR_t = Cvoid

# typedef clGetGLContextInfoKHR_t * clGetGLContextInfoKHR_fn
const clGetGLContextInfoKHR_fn = Ptr{clGetGLContextInfoKHR_t}

# typedef cl_mem CL_API_CALL clCreateFromGLBuffer_t ( cl_context context , cl_mem_flags flags , cl_GLuint bufobj , cl_int * errcode_ret )
const clCreateFromGLBuffer_t = Cvoid

# typedef clCreateFromGLBuffer_t * clCreateFromGLBuffer_fn
const clCreateFromGLBuffer_fn = Ptr{clCreateFromGLBuffer_t}

@checked function clGetGLContextInfoKHR(properties, param_name, param_value_size,
                                        param_value, param_value_size_ret)
    @ccall libopencl.clGetGLContextInfoKHR(properties::Ptr{cl_context_properties},
                                           param_name::cl_gl_context_info,
                                           param_value_size::Csize_t,
                                           param_value::Ptr{Cvoid},
                                           param_value_size_ret::Ptr{Csize_t})::cl_int
end

function clCreateFromGLBuffer(context, flags, bufobj, errcode_ret)
    @ccall libopencl.clCreateFromGLBuffer(context::cl_context, flags::cl_mem_flags,
                                          bufobj::cl_GLuint,
                                          errcode_ret::Ptr{cl_int})::cl_mem
end

# typedef cl_mem CL_API_CALL clCreateFromGLTexture_t ( cl_context context , cl_mem_flags flags , cl_GLenum target , cl_GLint miplevel , cl_GLuint texture , cl_int * errcode_ret )
const clCreateFromGLTexture_t = Cvoid

# typedef clCreateFromGLTexture_t * clCreateFromGLTexture_fn
const clCreateFromGLTexture_fn = Ptr{clCreateFromGLTexture_t}

function clCreateFromGLTexture(context, flags, target, miplevel, texture, errcode_ret)
    @ccall libopencl.clCreateFromGLTexture(context::cl_context, flags::cl_mem_flags,
                                           target::cl_GLenum, miplevel::cl_GLint,
                                           texture::cl_GLuint,
                                           errcode_ret::Ptr{cl_int})::cl_mem
end

# typedef cl_mem CL_API_CALL clCreateFromGLRenderbuffer_t ( cl_context context , cl_mem_flags flags , cl_GLuint renderbuffer , cl_int * errcode_ret )
const clCreateFromGLRenderbuffer_t = Cvoid

# typedef clCreateFromGLRenderbuffer_t * clCreateFromGLRenderbuffer_fn
const clCreateFromGLRenderbuffer_fn = Ptr{clCreateFromGLRenderbuffer_t}

# typedef cl_int CL_API_CALL clGetGLObjectInfo_t ( cl_mem memobj , cl_gl_object_type * gl_object_type , cl_GLuint * gl_object_name )
const clGetGLObjectInfo_t = Cvoid

# typedef clGetGLObjectInfo_t * clGetGLObjectInfo_fn
const clGetGLObjectInfo_fn = Ptr{clGetGLObjectInfo_t}

# typedef cl_int CL_API_CALL clGetGLTextureInfo_t ( cl_mem memobj , cl_gl_texture_info param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetGLTextureInfo_t = Cvoid

# typedef clGetGLTextureInfo_t * clGetGLTextureInfo_fn
const clGetGLTextureInfo_fn = Ptr{clGetGLTextureInfo_t}

# typedef cl_int CL_API_CALL clEnqueueAcquireGLObjects_t ( cl_command_queue command_queue , cl_uint num_objects , const cl_mem * mem_objects , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueAcquireGLObjects_t = Cvoid

# typedef clEnqueueAcquireGLObjects_t * clEnqueueAcquireGLObjects_fn
const clEnqueueAcquireGLObjects_fn = Ptr{clEnqueueAcquireGLObjects_t}

# typedef cl_int CL_API_CALL clEnqueueReleaseGLObjects_t ( cl_command_queue command_queue , cl_uint num_objects , const cl_mem * mem_objects , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueReleaseGLObjects_t = Cvoid

# typedef clEnqueueReleaseGLObjects_t * clEnqueueReleaseGLObjects_fn
const clEnqueueReleaseGLObjects_fn = Ptr{clEnqueueReleaseGLObjects_t}

function clCreateFromGLRenderbuffer(context, flags, renderbuffer, errcode_ret)
    @ccall libopencl.clCreateFromGLRenderbuffer(context::cl_context, flags::cl_mem_flags,
                                                renderbuffer::cl_GLuint,
                                                errcode_ret::Ptr{cl_int})::cl_mem
end

@checked function clGetGLObjectInfo(memobj, gl_object_type, gl_object_name)
    @ccall libopencl.clGetGLObjectInfo(memobj::cl_mem,
                                       gl_object_type::Ptr{cl_gl_object_type},
                                       gl_object_name::Ptr{cl_GLuint})::cl_int
end

@checked function clGetGLTextureInfo(memobj, param_name, param_value_size, param_value,
                                     param_value_size_ret)
    @ccall libopencl.clGetGLTextureInfo(memobj::cl_mem, param_name::cl_gl_texture_info,
                                        param_value_size::Csize_t, param_value::Ptr{Cvoid},
                                        param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clEnqueueAcquireGLObjects(command_queue, num_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueAcquireGLObjects(command_queue::cl_command_queue,
                                               num_objects::cl_uint,
                                               mem_objects::Ptr{cl_mem},
                                               num_events_in_wait_list::cl_uint,
                                               event_wait_list::Ptr{cl_event},
                                               event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueReleaseGLObjects(command_queue, num_objects, mem_objects,
                                            num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueReleaseGLObjects(command_queue::cl_command_queue,
                                               num_objects::cl_uint,
                                               mem_objects::Ptr{cl_mem},
                                               num_events_in_wait_list::cl_uint,
                                               event_wait_list::Ptr{cl_event},
                                               event::Ptr{cl_event})::cl_int
end

# typedef cl_mem CL_API_CALL clCreateFromGLTexture2D_t ( cl_context context , cl_mem_flags flags , cl_GLenum target , cl_GLint miplevel , cl_GLuint texture , cl_int * errcode_ret )
const clCreateFromGLTexture2D_t = Cvoid

# typedef clCreateFromGLTexture2D_t * clCreateFromGLTexture2D_fn
const clCreateFromGLTexture2D_fn = Ptr{clCreateFromGLTexture2D_t}

# typedef cl_mem CL_API_CALL clCreateFromGLTexture3D_t ( cl_context context , cl_mem_flags flags , cl_GLenum target , cl_GLint miplevel , cl_GLuint texture , cl_int * errcode_ret )
const clCreateFromGLTexture3D_t = Cvoid

# typedef clCreateFromGLTexture3D_t * clCreateFromGLTexture3D_fn
const clCreateFromGLTexture3D_fn = Ptr{clCreateFromGLTexture3D_t}

function clCreateFromGLTexture2D(context, flags, target, miplevel, texture, errcode_ret)
    @ccall libopencl.clCreateFromGLTexture2D(context::cl_context, flags::cl_mem_flags,
                                             target::cl_GLenum, miplevel::cl_GLint,
                                             texture::cl_GLuint,
                                             errcode_ret::Ptr{cl_int})::cl_mem
end

function clCreateFromGLTexture3D(context, flags, target, miplevel, texture, errcode_ret)
    @ccall libopencl.clCreateFromGLTexture3D(context::cl_context, flags::cl_mem_flags,
                                             target::cl_GLenum, miplevel::cl_GLint,
                                             texture::cl_GLuint,
                                             errcode_ret::Ptr{cl_int})::cl_mem
end

mutable struct __GLsync end

const cl_GLsync = Ptr{__GLsync}

# typedef cl_event CL_API_CALL clCreateEventFromGLsyncKHR_t ( cl_context context , cl_GLsync sync , cl_int * errcode_ret )
const clCreateEventFromGLsyncKHR_t = Cvoid

# typedef clCreateEventFromGLsyncKHR_t * clCreateEventFromGLsyncKHR_fn
const clCreateEventFromGLsyncKHR_fn = Ptr{clCreateEventFromGLsyncKHR_t}

function clCreateEventFromGLsyncKHR(context, sync, errcode_ret)
    @ccall libopencl.clCreateEventFromGLsyncKHR(context::cl_context, sync::cl_GLsync,
                                                errcode_ret::Ptr{cl_int})::cl_event
end

# typedef cl_int CL_API_CALL clGetSupportedGLTextureFormatsINTEL_t ( cl_context context , cl_mem_flags flags , cl_mem_object_type image_type , cl_uint num_entries , cl_GLenum * gl_formats , cl_uint * num_texture_formats )
const clGetSupportedGLTextureFormatsINTEL_t = Cvoid

# typedef clGetSupportedGLTextureFormatsINTEL_t * clGetSupportedGLTextureFormatsINTEL_fn
const clGetSupportedGLTextureFormatsINTEL_fn = Ptr{clGetSupportedGLTextureFormatsINTEL_t}

@checked function clGetSupportedGLTextureFormatsINTEL(context, flags, image_type,
                                                      num_entries, gl_formats,
                                                      num_texture_formats)
    @ccall libopencl.clGetSupportedGLTextureFormatsINTEL(context::cl_context,
                                                         flags::cl_mem_flags,
                                                         image_type::cl_mem_object_type,
                                                         num_entries::cl_uint,
                                                         gl_formats::Ptr{cl_GLenum},
                                                         num_texture_formats::Ptr{cl_uint})::cl_int
end

const cl_device_partition_property_ext = cl_ulong

const cl_device_command_buffer_capabilities_khr = cl_bitfield

mutable struct _cl_command_buffer_khr end

const cl_command_buffer_khr = Ptr{_cl_command_buffer_khr}

const cl_sync_point_khr = cl_uint

const cl_command_buffer_info_khr = cl_uint

const cl_command_buffer_state_khr = cl_uint

const cl_command_buffer_properties_khr = cl_properties

const cl_command_buffer_flags_khr = cl_bitfield

const cl_ndrange_kernel_command_properties_khr = cl_properties

mutable struct _cl_mutable_command_khr end

const cl_mutable_command_khr = Ptr{_cl_mutable_command_khr}

# typedef cl_command_buffer_khr CL_API_CALL clCreateCommandBufferKHR_t ( cl_uint num_queues , const cl_command_queue * queues , const cl_command_buffer_properties_khr * properties , cl_int * errcode_ret )
const clCreateCommandBufferKHR_t = Cvoid

# typedef clCreateCommandBufferKHR_t * clCreateCommandBufferKHR_fn
const clCreateCommandBufferKHR_fn = Ptr{clCreateCommandBufferKHR_t}

# typedef cl_int CL_API_CALL clFinalizeCommandBufferKHR_t ( cl_command_buffer_khr command_buffer )
const clFinalizeCommandBufferKHR_t = Cvoid

# typedef clFinalizeCommandBufferKHR_t * clFinalizeCommandBufferKHR_fn
const clFinalizeCommandBufferKHR_fn = Ptr{clFinalizeCommandBufferKHR_t}

# typedef cl_int CL_API_CALL clRetainCommandBufferKHR_t ( cl_command_buffer_khr command_buffer )
const clRetainCommandBufferKHR_t = Cvoid

# typedef clRetainCommandBufferKHR_t * clRetainCommandBufferKHR_fn
const clRetainCommandBufferKHR_fn = Ptr{clRetainCommandBufferKHR_t}

# typedef cl_int CL_API_CALL clReleaseCommandBufferKHR_t ( cl_command_buffer_khr command_buffer )
const clReleaseCommandBufferKHR_t = Cvoid

# typedef clReleaseCommandBufferKHR_t * clReleaseCommandBufferKHR_fn
const clReleaseCommandBufferKHR_fn = Ptr{clReleaseCommandBufferKHR_t}

# typedef cl_int CL_API_CALL clEnqueueCommandBufferKHR_t ( cl_uint num_queues , cl_command_queue * queues , cl_command_buffer_khr command_buffer , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueCommandBufferKHR_t = Cvoid

# typedef clEnqueueCommandBufferKHR_t * clEnqueueCommandBufferKHR_fn
const clEnqueueCommandBufferKHR_fn = Ptr{clEnqueueCommandBufferKHR_t}

# typedef cl_int CL_API_CALL clCommandBarrierWithWaitListKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandBarrierWithWaitListKHR_t = Cvoid

# typedef clCommandBarrierWithWaitListKHR_t * clCommandBarrierWithWaitListKHR_fn
const clCommandBarrierWithWaitListKHR_fn = Ptr{clCommandBarrierWithWaitListKHR_t}

# typedef cl_int CL_API_CALL clCommandCopyBufferKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , cl_mem src_buffer , cl_mem dst_buffer , size_t src_offset , size_t dst_offset , size_t size , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandCopyBufferKHR_t = Cvoid

# typedef clCommandCopyBufferKHR_t * clCommandCopyBufferKHR_fn
const clCommandCopyBufferKHR_fn = Ptr{clCommandCopyBufferKHR_t}

# typedef cl_int CL_API_CALL clCommandCopyBufferRectKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , cl_mem src_buffer , cl_mem dst_buffer , const size_t * src_origin , const size_t * dst_origin , const size_t * region , size_t src_row_pitch , size_t src_slice_pitch , size_t dst_row_pitch , size_t dst_slice_pitch , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandCopyBufferRectKHR_t = Cvoid

# typedef clCommandCopyBufferRectKHR_t * clCommandCopyBufferRectKHR_fn
const clCommandCopyBufferRectKHR_fn = Ptr{clCommandCopyBufferRectKHR_t}

# typedef cl_int CL_API_CALL clCommandCopyBufferToImageKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , cl_mem src_buffer , cl_mem dst_image , size_t src_offset , const size_t * dst_origin , const size_t * region , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandCopyBufferToImageKHR_t = Cvoid

# typedef clCommandCopyBufferToImageKHR_t * clCommandCopyBufferToImageKHR_fn
const clCommandCopyBufferToImageKHR_fn = Ptr{clCommandCopyBufferToImageKHR_t}

# typedef cl_int CL_API_CALL clCommandCopyImageKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , cl_mem src_image , cl_mem dst_image , const size_t * src_origin , const size_t * dst_origin , const size_t * region , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandCopyImageKHR_t = Cvoid

# typedef clCommandCopyImageKHR_t * clCommandCopyImageKHR_fn
const clCommandCopyImageKHR_fn = Ptr{clCommandCopyImageKHR_t}

# typedef cl_int CL_API_CALL clCommandCopyImageToBufferKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , cl_mem src_image , cl_mem dst_buffer , const size_t * src_origin , const size_t * region , size_t dst_offset , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandCopyImageToBufferKHR_t = Cvoid

# typedef clCommandCopyImageToBufferKHR_t * clCommandCopyImageToBufferKHR_fn
const clCommandCopyImageToBufferKHR_fn = Ptr{clCommandCopyImageToBufferKHR_t}

# typedef cl_int CL_API_CALL clCommandFillBufferKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , cl_mem buffer , const void * pattern , size_t pattern_size , size_t offset , size_t size , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandFillBufferKHR_t = Cvoid

# typedef clCommandFillBufferKHR_t * clCommandFillBufferKHR_fn
const clCommandFillBufferKHR_fn = Ptr{clCommandFillBufferKHR_t}

# typedef cl_int CL_API_CALL clCommandFillImageKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , cl_mem image , const void * fill_color , const size_t * origin , const size_t * region , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandFillImageKHR_t = Cvoid

# typedef clCommandFillImageKHR_t * clCommandFillImageKHR_fn
const clCommandFillImageKHR_fn = Ptr{clCommandFillImageKHR_t}

# typedef cl_int CL_API_CALL clCommandNDRangeKernelKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , const cl_ndrange_kernel_command_properties_khr * properties , cl_kernel kernel , cl_uint work_dim , const size_t * global_work_offset , const size_t * global_work_size , const size_t * local_work_size , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandNDRangeKernelKHR_t = Cvoid

# typedef clCommandNDRangeKernelKHR_t * clCommandNDRangeKernelKHR_fn
const clCommandNDRangeKernelKHR_fn = Ptr{clCommandNDRangeKernelKHR_t}

# typedef cl_int CL_API_CALL clGetCommandBufferInfoKHR_t ( cl_command_buffer_khr command_buffer , cl_command_buffer_info_khr param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetCommandBufferInfoKHR_t = Cvoid

# typedef clGetCommandBufferInfoKHR_t * clGetCommandBufferInfoKHR_fn
const clGetCommandBufferInfoKHR_fn = Ptr{clGetCommandBufferInfoKHR_t}

function clCreateCommandBufferKHR(num_queues, queues, properties, errcode_ret)
    @ccall libopencl.clCreateCommandBufferKHR(num_queues::cl_uint,
                                              queues::Ptr{cl_command_queue},
                                              properties::Ptr{cl_command_buffer_properties_khr},
                                              errcode_ret::Ptr{cl_int})::cl_command_buffer_khr
end

@checked function clFinalizeCommandBufferKHR(command_buffer)
    @ccall libopencl.clFinalizeCommandBufferKHR(command_buffer::cl_command_buffer_khr)::cl_int
end

@checked function clRetainCommandBufferKHR(command_buffer)
    @ccall libopencl.clRetainCommandBufferKHR(command_buffer::cl_command_buffer_khr)::cl_int
end

@checked function clReleaseCommandBufferKHR(command_buffer)
    @ccall libopencl.clReleaseCommandBufferKHR(command_buffer::cl_command_buffer_khr)::cl_int
end

@checked function clEnqueueCommandBufferKHR(num_queues, queues, command_buffer,
                                            num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueCommandBufferKHR(num_queues::cl_uint,
                                               queues::Ptr{cl_command_queue},
                                               command_buffer::cl_command_buffer_khr,
                                               num_events_in_wait_list::cl_uint,
                                               event_wait_list::Ptr{cl_event},
                                               event::Ptr{cl_event})::cl_int
end

@checked function clCommandBarrierWithWaitListKHR(command_buffer, command_queue,
                                                  num_sync_points_in_wait_list,
                                                  sync_point_wait_list, sync_point,
                                                  mutable_handle)
    @ccall libopencl.clCommandBarrierWithWaitListKHR(command_buffer::cl_command_buffer_khr,
                                                     command_queue::cl_command_queue,
                                                     num_sync_points_in_wait_list::cl_uint,
                                                     sync_point_wait_list::Ptr{cl_sync_point_khr},
                                                     sync_point::Ptr{cl_sync_point_khr},
                                                     mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandCopyBufferKHR(command_buffer, command_queue, src_buffer,
                                         dst_buffer, src_offset, dst_offset, size,
                                         num_sync_points_in_wait_list, sync_point_wait_list,
                                         sync_point, mutable_handle)
    @ccall libopencl.clCommandCopyBufferKHR(command_buffer::cl_command_buffer_khr,
                                            command_queue::cl_command_queue,
                                            src_buffer::cl_mem, dst_buffer::cl_mem,
                                            src_offset::Csize_t, dst_offset::Csize_t,
                                            size::Csize_t,
                                            num_sync_points_in_wait_list::cl_uint,
                                            sync_point_wait_list::Ptr{cl_sync_point_khr},
                                            sync_point::Ptr{cl_sync_point_khr},
                                            mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandCopyBufferRectKHR(command_buffer, command_queue, src_buffer,
                                             dst_buffer, src_origin, dst_origin, region,
                                             src_row_pitch, src_slice_pitch, dst_row_pitch,
                                             dst_slice_pitch, num_sync_points_in_wait_list,
                                             sync_point_wait_list, sync_point,
                                             mutable_handle)
    @ccall libopencl.clCommandCopyBufferRectKHR(command_buffer::cl_command_buffer_khr,
                                                command_queue::cl_command_queue,
                                                src_buffer::cl_mem, dst_buffer::cl_mem,
                                                src_origin::Ptr{Csize_t},
                                                dst_origin::Ptr{Csize_t},
                                                region::Ptr{Csize_t},
                                                src_row_pitch::Csize_t,
                                                src_slice_pitch::Csize_t,
                                                dst_row_pitch::Csize_t,
                                                dst_slice_pitch::Csize_t,
                                                num_sync_points_in_wait_list::cl_uint,
                                                sync_point_wait_list::Ptr{cl_sync_point_khr},
                                                sync_point::Ptr{cl_sync_point_khr},
                                                mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandCopyBufferToImageKHR(command_buffer, command_queue, src_buffer,
                                                dst_image, src_offset, dst_origin, region,
                                                num_sync_points_in_wait_list,
                                                sync_point_wait_list, sync_point,
                                                mutable_handle)
    @ccall libopencl.clCommandCopyBufferToImageKHR(command_buffer::cl_command_buffer_khr,
                                                   command_queue::cl_command_queue,
                                                   src_buffer::cl_mem, dst_image::cl_mem,
                                                   src_offset::Csize_t,
                                                   dst_origin::Ptr{Csize_t},
                                                   region::Ptr{Csize_t},
                                                   num_sync_points_in_wait_list::cl_uint,
                                                   sync_point_wait_list::Ptr{cl_sync_point_khr},
                                                   sync_point::Ptr{cl_sync_point_khr},
                                                   mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandCopyImageKHR(command_buffer, command_queue, src_image, dst_image,
                                        src_origin, dst_origin, region,
                                        num_sync_points_in_wait_list, sync_point_wait_list,
                                        sync_point, mutable_handle)
    @ccall libopencl.clCommandCopyImageKHR(command_buffer::cl_command_buffer_khr,
                                           command_queue::cl_command_queue,
                                           src_image::cl_mem, dst_image::cl_mem,
                                           src_origin::Ptr{Csize_t},
                                           dst_origin::Ptr{Csize_t}, region::Ptr{Csize_t},
                                           num_sync_points_in_wait_list::cl_uint,
                                           sync_point_wait_list::Ptr{cl_sync_point_khr},
                                           sync_point::Ptr{cl_sync_point_khr},
                                           mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandCopyImageToBufferKHR(command_buffer, command_queue, src_image,
                                                dst_buffer, src_origin, region, dst_offset,
                                                num_sync_points_in_wait_list,
                                                sync_point_wait_list, sync_point,
                                                mutable_handle)
    @ccall libopencl.clCommandCopyImageToBufferKHR(command_buffer::cl_command_buffer_khr,
                                                   command_queue::cl_command_queue,
                                                   src_image::cl_mem, dst_buffer::cl_mem,
                                                   src_origin::Ptr{Csize_t},
                                                   region::Ptr{Csize_t},
                                                   dst_offset::Csize_t,
                                                   num_sync_points_in_wait_list::cl_uint,
                                                   sync_point_wait_list::Ptr{cl_sync_point_khr},
                                                   sync_point::Ptr{cl_sync_point_khr},
                                                   mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandFillBufferKHR(command_buffer, command_queue, buffer, pattern,
                                         pattern_size, offset, size,
                                         num_sync_points_in_wait_list, sync_point_wait_list,
                                         sync_point, mutable_handle)
    @ccall libopencl.clCommandFillBufferKHR(command_buffer::cl_command_buffer_khr,
                                            command_queue::cl_command_queue, buffer::cl_mem,
                                            pattern::Ptr{Cvoid}, pattern_size::Csize_t,
                                            offset::Csize_t, size::Csize_t,
                                            num_sync_points_in_wait_list::cl_uint,
                                            sync_point_wait_list::Ptr{cl_sync_point_khr},
                                            sync_point::Ptr{cl_sync_point_khr},
                                            mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandFillImageKHR(command_buffer, command_queue, image, fill_color,
                                        origin, region, num_sync_points_in_wait_list,
                                        sync_point_wait_list, sync_point, mutable_handle)
    @ccall libopencl.clCommandFillImageKHR(command_buffer::cl_command_buffer_khr,
                                           command_queue::cl_command_queue, image::cl_mem,
                                           fill_color::Ptr{Cvoid}, origin::Ptr{Csize_t},
                                           region::Ptr{Csize_t},
                                           num_sync_points_in_wait_list::cl_uint,
                                           sync_point_wait_list::Ptr{cl_sync_point_khr},
                                           sync_point::Ptr{cl_sync_point_khr},
                                           mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandNDRangeKernelKHR(command_buffer, command_queue, properties,
                                            kernel, work_dim, global_work_offset,
                                            global_work_size, local_work_size,
                                            num_sync_points_in_wait_list,
                                            sync_point_wait_list, sync_point,
                                            mutable_handle)
    @ccall libopencl.clCommandNDRangeKernelKHR(command_buffer::cl_command_buffer_khr,
                                               command_queue::cl_command_queue,
                                               properties::Ptr{cl_ndrange_kernel_command_properties_khr},
                                               kernel::cl_kernel, work_dim::cl_uint,
                                               global_work_offset::Ptr{Csize_t},
                                               global_work_size::Ptr{Csize_t},
                                               local_work_size::Ptr{Csize_t},
                                               num_sync_points_in_wait_list::cl_uint,
                                               sync_point_wait_list::Ptr{cl_sync_point_khr},
                                               sync_point::Ptr{cl_sync_point_khr},
                                               mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clGetCommandBufferInfoKHR(command_buffer, param_name, param_value_size,
                                            param_value, param_value_size_ret)
    @ccall libopencl.clGetCommandBufferInfoKHR(command_buffer::cl_command_buffer_khr,
                                               param_name::cl_command_buffer_info_khr,
                                               param_value_size::Csize_t,
                                               param_value::Ptr{Cvoid},
                                               param_value_size_ret::Ptr{Csize_t})::cl_int
end

# typedef cl_int CL_API_CALL clCommandSVMMemcpyKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , void * dst_ptr , const void * src_ptr , size_t size , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandSVMMemcpyKHR_t = Cvoid

# typedef clCommandSVMMemcpyKHR_t * clCommandSVMMemcpyKHR_fn
const clCommandSVMMemcpyKHR_fn = Ptr{clCommandSVMMemcpyKHR_t}

# typedef cl_int CL_API_CALL clCommandSVMMemFillKHR_t ( cl_command_buffer_khr command_buffer , cl_command_queue command_queue , void * svm_ptr , const void * pattern , size_t pattern_size , size_t size , cl_uint num_sync_points_in_wait_list , const cl_sync_point_khr * sync_point_wait_list , cl_sync_point_khr * sync_point , cl_mutable_command_khr * mutable_handle )
const clCommandSVMMemFillKHR_t = Cvoid

# typedef clCommandSVMMemFillKHR_t * clCommandSVMMemFillKHR_fn
const clCommandSVMMemFillKHR_fn = Ptr{clCommandSVMMemFillKHR_t}

@checked function clCommandSVMMemcpyKHR(command_buffer, command_queue, dst_ptr, src_ptr,
                                        size, num_sync_points_in_wait_list,
                                        sync_point_wait_list, sync_point, mutable_handle)
    @ccall libopencl.clCommandSVMMemcpyKHR(command_buffer::cl_command_buffer_khr,
                                           command_queue::cl_command_queue,
                                           dst_ptr::Ptr{Cvoid}, src_ptr::Ptr{Cvoid},
                                           size::Csize_t,
                                           num_sync_points_in_wait_list::cl_uint,
                                           sync_point_wait_list::Ptr{cl_sync_point_khr},
                                           sync_point::Ptr{cl_sync_point_khr},
                                           mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

@checked function clCommandSVMMemFillKHR(command_buffer, command_queue, svm_ptr, pattern,
                                         pattern_size, size, num_sync_points_in_wait_list,
                                         sync_point_wait_list, sync_point, mutable_handle)
    @ccall libopencl.clCommandSVMMemFillKHR(command_buffer::cl_command_buffer_khr,
                                            command_queue::cl_command_queue,
                                            svm_ptr::Ptr{Cvoid}, pattern::Ptr{Cvoid},
                                            pattern_size::Csize_t, size::Csize_t,
                                            num_sync_points_in_wait_list::cl_uint,
                                            sync_point_wait_list::Ptr{cl_sync_point_khr},
                                            sync_point::Ptr{cl_sync_point_khr},
                                            mutable_handle::Ptr{cl_mutable_command_khr})::cl_int
end

const cl_platform_command_buffer_capabilities_khr = cl_bitfield

# typedef cl_command_buffer_khr CL_API_CALL clRemapCommandBufferKHR_t ( cl_command_buffer_khr command_buffer , cl_bool automatic , cl_uint num_queues , const cl_command_queue * queues , cl_uint num_handles , const cl_mutable_command_khr * handles , cl_mutable_command_khr * handles_ret , cl_int * errcode_ret )
const clRemapCommandBufferKHR_t = Cvoid

# typedef clRemapCommandBufferKHR_t * clRemapCommandBufferKHR_fn
const clRemapCommandBufferKHR_fn = Ptr{clRemapCommandBufferKHR_t}

function clRemapCommandBufferKHR(command_buffer, automatic, num_queues, queues, num_handles,
                                 handles, handles_ret, errcode_ret)
    @ccall libopencl.clRemapCommandBufferKHR(command_buffer::cl_command_buffer_khr,
                                             automatic::cl_bool, num_queues::cl_uint,
                                             queues::Ptr{cl_command_queue},
                                             num_handles::cl_uint,
                                             handles::Ptr{cl_mutable_command_khr},
                                             handles_ret::Ptr{cl_mutable_command_khr},
                                             errcode_ret::Ptr{cl_int})::cl_command_buffer_khr
end

const cl_command_buffer_structure_type_khr = cl_uint

const cl_mutable_dispatch_fields_khr = cl_bitfield

const cl_mutable_command_info_khr = cl_uint

struct _cl_mutable_dispatch_arg_khr
    arg_index::cl_uint
    arg_size::Csize_t
    arg_value::Ptr{Cvoid}
end

const cl_mutable_dispatch_arg_khr = _cl_mutable_dispatch_arg_khr

struct _cl_mutable_dispatch_exec_info_khr
    param_name::cl_uint
    param_value_size::Csize_t
    param_value::Ptr{Cvoid}
end

const cl_mutable_dispatch_exec_info_khr = _cl_mutable_dispatch_exec_info_khr

struct _cl_mutable_dispatch_config_khr
    type::cl_command_buffer_structure_type_khr
    next::Ptr{Cvoid}
    command::cl_mutable_command_khr
    num_args::cl_uint
    num_svm_args::cl_uint
    num_exec_infos::cl_uint
    work_dim::cl_uint
    arg_list::Ptr{cl_mutable_dispatch_arg_khr}
    arg_svm_list::Ptr{cl_mutable_dispatch_arg_khr}
    exec_info_list::Ptr{cl_mutable_dispatch_exec_info_khr}
    global_work_offset::Ptr{Csize_t}
    global_work_size::Ptr{Csize_t}
    local_work_size::Ptr{Csize_t}
end

const cl_mutable_dispatch_config_khr = _cl_mutable_dispatch_config_khr

struct _cl_mutable_base_config_khr
    type::cl_command_buffer_structure_type_khr
    next::Ptr{Cvoid}
    num_mutable_dispatch::cl_uint
    mutable_dispatch_list::Ptr{cl_mutable_dispatch_config_khr}
end

const cl_mutable_base_config_khr = _cl_mutable_base_config_khr

const cl_mutable_dispatch_asserts_khr = cl_bitfield

# typedef cl_int CL_API_CALL clUpdateMutableCommandsKHR_t ( cl_command_buffer_khr command_buffer , const cl_mutable_base_config_khr * mutable_config )
const clUpdateMutableCommandsKHR_t = Cvoid

# typedef clUpdateMutableCommandsKHR_t * clUpdateMutableCommandsKHR_fn
const clUpdateMutableCommandsKHR_fn = Ptr{clUpdateMutableCommandsKHR_t}

# typedef cl_int CL_API_CALL clGetMutableCommandInfoKHR_t ( cl_mutable_command_khr command , cl_mutable_command_info_khr param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetMutableCommandInfoKHR_t = Cvoid

# typedef clGetMutableCommandInfoKHR_t * clGetMutableCommandInfoKHR_fn
const clGetMutableCommandInfoKHR_fn = Ptr{clGetMutableCommandInfoKHR_t}

@checked function clUpdateMutableCommandsKHR(command_buffer, mutable_config)
    @ccall libopencl.clUpdateMutableCommandsKHR(command_buffer::cl_command_buffer_khr,
                                                mutable_config::Ptr{cl_mutable_base_config_khr})::cl_int
end

@checked function clGetMutableCommandInfoKHR(command, param_name, param_value_size,
                                             param_value, param_value_size_ret)
    @ccall libopencl.clGetMutableCommandInfoKHR(command::cl_mutable_command_khr,
                                                param_name::cl_mutable_command_info_khr,
                                                param_value_size::Csize_t,
                                                param_value::Ptr{Cvoid},
                                                param_value_size_ret::Ptr{Csize_t})::cl_int
end

# typedef cl_int CL_API_CALL clSetMemObjectDestructorAPPLE_t ( cl_mem memobj , void ( CL_CALLBACK * pfn_notify ) ( cl_mem memobj , void * user_data ) , void * user_data )
const clSetMemObjectDestructorAPPLE_t = Cvoid

# typedef clSetMemObjectDestructorAPPLE_t * clSetMemObjectDestructorAPPLE_fn
const clSetMemObjectDestructorAPPLE_fn = Ptr{clSetMemObjectDestructorAPPLE_t}

@checked function clSetMemObjectDestructorAPPLE(memobj, pfn_notify, user_data)
    @ccall libopencl.clSetMemObjectDestructorAPPLE(memobj::cl_mem, pfn_notify::Ptr{Cvoid},
                                                   user_data::Ptr{Cvoid})::cl_int
end

# typedef void CL_API_CALL clLogMessagesToSystemLogAPPLE_t ( const char * errstr , const void * private_info , size_t cb , void * user_data )
const clLogMessagesToSystemLogAPPLE_t = Cvoid

# typedef clLogMessagesToSystemLogAPPLE_t * clLogMessagesToSystemLogAPPLE_fn
const clLogMessagesToSystemLogAPPLE_fn = Ptr{clLogMessagesToSystemLogAPPLE_t}

# typedef void CL_API_CALL clLogMessagesToStdoutAPPLE_t ( const char * errstr , const void * private_info , size_t cb , void * user_data )
const clLogMessagesToStdoutAPPLE_t = Cvoid

# typedef clLogMessagesToStdoutAPPLE_t * clLogMessagesToStdoutAPPLE_fn
const clLogMessagesToStdoutAPPLE_fn = Ptr{clLogMessagesToStdoutAPPLE_t}

# typedef void CL_API_CALL clLogMessagesToStderrAPPLE_t ( const char * errstr , const void * private_info , size_t cb , void * user_data )
const clLogMessagesToStderrAPPLE_t = Cvoid

# typedef clLogMessagesToStderrAPPLE_t * clLogMessagesToStderrAPPLE_fn
const clLogMessagesToStderrAPPLE_fn = Ptr{clLogMessagesToStderrAPPLE_t}

function clLogMessagesToSystemLogAPPLE(errstr, private_info, cb, user_data)
    @ccall libopencl.clLogMessagesToSystemLogAPPLE(errstr::Ptr{Cchar},
                                                   private_info::Ptr{Cvoid}, cb::Csize_t,
                                                   user_data::Ptr{Cvoid})::Cvoid
end

function clLogMessagesToStdoutAPPLE(errstr, private_info, cb, user_data)
    @ccall libopencl.clLogMessagesToStdoutAPPLE(errstr::Ptr{Cchar},
                                                private_info::Ptr{Cvoid}, cb::Csize_t,
                                                user_data::Ptr{Cvoid})::Cvoid
end

function clLogMessagesToStderrAPPLE(errstr, private_info, cb, user_data)
    @ccall libopencl.clLogMessagesToStderrAPPLE(errstr::Ptr{Cchar},
                                                private_info::Ptr{Cvoid}, cb::Csize_t,
                                                user_data::Ptr{Cvoid})::Cvoid
end

# typedef cl_int CL_API_CALL clIcdGetPlatformIDsKHR_t ( cl_uint num_entries , cl_platform_id * platforms , cl_uint * num_platforms )
const clIcdGetPlatformIDsKHR_t = Cvoid

# typedef clIcdGetPlatformIDsKHR_t * clIcdGetPlatformIDsKHR_fn
const clIcdGetPlatformIDsKHR_fn = Ptr{clIcdGetPlatformIDsKHR_t}

@checked function clIcdGetPlatformIDsKHR(num_entries, platforms, num_platforms)
    @ccall libopencl.clIcdGetPlatformIDsKHR(num_entries::cl_uint,
                                            platforms::Ptr{cl_platform_id},
                                            num_platforms::Ptr{cl_uint})::cl_int
end

# typedef cl_program CL_API_CALL clCreateProgramWithILKHR_t ( cl_context context , const void * il , size_t length , cl_int * errcode_ret )
const clCreateProgramWithILKHR_t = Cvoid

# typedef clCreateProgramWithILKHR_t * clCreateProgramWithILKHR_fn
const clCreateProgramWithILKHR_fn = Ptr{clCreateProgramWithILKHR_t}

function clCreateProgramWithILKHR(context, il, length, errcode_ret)
    @ccall libopencl.clCreateProgramWithILKHR(context::cl_context, il::Ptr{Cvoid},
                                              length::Csize_t,
                                              errcode_ret::Ptr{cl_int})::cl_program
end

const cl_context_memory_initialize_khr = cl_bitfield

const cl_device_terminate_capability_khr = cl_bitfield

# typedef cl_int CL_API_CALL clTerminateContextKHR_t ( cl_context context )
const clTerminateContextKHR_t = Cvoid

# typedef clTerminateContextKHR_t * clTerminateContextKHR_fn
const clTerminateContextKHR_fn = Ptr{clTerminateContextKHR_t}

@checked function clTerminateContextKHR(context)
    @ccall libopencl.clTerminateContextKHR(context::cl_context)::cl_int
end

const cl_queue_properties_khr = cl_properties

# typedef cl_command_queue CL_API_CALL clCreateCommandQueueWithPropertiesKHR_t ( cl_context context , cl_device_id device , const cl_queue_properties_khr * properties , cl_int * errcode_ret )
const clCreateCommandQueueWithPropertiesKHR_t = Cvoid

# typedef clCreateCommandQueueWithPropertiesKHR_t * clCreateCommandQueueWithPropertiesKHR_fn
const clCreateCommandQueueWithPropertiesKHR_fn = Ptr{clCreateCommandQueueWithPropertiesKHR_t}

function clCreateCommandQueueWithPropertiesKHR(context, device, properties, errcode_ret)
    @ccall libopencl.clCreateCommandQueueWithPropertiesKHR(context::cl_context,
                                                           device::cl_device_id,
                                                           properties::Ptr{cl_queue_properties_khr},
                                                           errcode_ret::Ptr{cl_int})::cl_command_queue
end

# typedef cl_int CL_API_CALL clReleaseDeviceEXT_t ( cl_device_id device )
const clReleaseDeviceEXT_t = Cvoid

# typedef clReleaseDeviceEXT_t * clReleaseDeviceEXT_fn
const clReleaseDeviceEXT_fn = Ptr{clReleaseDeviceEXT_t}

# typedef cl_int CL_API_CALL clRetainDeviceEXT_t ( cl_device_id device )
const clRetainDeviceEXT_t = Cvoid

# typedef clRetainDeviceEXT_t * clRetainDeviceEXT_fn
const clRetainDeviceEXT_fn = Ptr{clRetainDeviceEXT_t}

# typedef cl_int CL_API_CALL clCreateSubDevicesEXT_t ( cl_device_id in_device , const cl_device_partition_property_ext * properties , cl_uint num_entries , cl_device_id * out_devices , cl_uint * num_devices )
const clCreateSubDevicesEXT_t = Cvoid

# typedef clCreateSubDevicesEXT_t * clCreateSubDevicesEXT_fn
const clCreateSubDevicesEXT_fn = Ptr{clCreateSubDevicesEXT_t}

@checked function clReleaseDeviceEXT(device)
    @ccall libopencl.clReleaseDeviceEXT(device::cl_device_id)::cl_int
end

@checked function clRetainDeviceEXT(device)
    @ccall libopencl.clRetainDeviceEXT(device::cl_device_id)::cl_int
end

@checked function clCreateSubDevicesEXT(in_device, properties, num_entries, out_devices,
                                        num_devices)
    @ccall libopencl.clCreateSubDevicesEXT(in_device::cl_device_id,
                                           properties::Ptr{cl_device_partition_property_ext},
                                           num_entries::cl_uint,
                                           out_devices::Ptr{cl_device_id},
                                           num_devices::Ptr{cl_uint})::cl_int
end

const cl_mem_migration_flags_ext = cl_bitfield

# typedef cl_int CL_API_CALL clEnqueueMigrateMemObjectEXT_t ( cl_command_queue command_queue , cl_uint num_mem_objects , const cl_mem * mem_objects , cl_mem_migration_flags_ext flags , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueMigrateMemObjectEXT_t = Cvoid

# typedef clEnqueueMigrateMemObjectEXT_t * clEnqueueMigrateMemObjectEXT_fn
const clEnqueueMigrateMemObjectEXT_fn = Ptr{clEnqueueMigrateMemObjectEXT_t}

@checked function clEnqueueMigrateMemObjectEXT(command_queue, num_mem_objects, mem_objects,
                                               flags, num_events_in_wait_list,
                                               event_wait_list, event)
    @ccall libopencl.clEnqueueMigrateMemObjectEXT(command_queue::cl_command_queue,
                                                  num_mem_objects::cl_uint,
                                                  mem_objects::Ptr{cl_mem},
                                                  flags::cl_mem_migration_flags_ext,
                                                  num_events_in_wait_list::cl_uint,
                                                  event_wait_list::Ptr{cl_event},
                                                  event::Ptr{cl_event})::cl_int
end

const cl_image_pitch_info_qcom = cl_uint

struct _cl_mem_ext_host_ptr
    allocation_type::cl_uint
    host_cache_policy::cl_uint
end

const cl_mem_ext_host_ptr = _cl_mem_ext_host_ptr

# typedef cl_int CL_API_CALL clGetDeviceImageInfoQCOM_t ( cl_device_id device , size_t image_width , size_t image_height , const cl_image_format * image_format , cl_image_pitch_info_qcom param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetDeviceImageInfoQCOM_t = Cvoid

# typedef clGetDeviceImageInfoQCOM_t * clGetDeviceImageInfoQCOM_fn
const clGetDeviceImageInfoQCOM_fn = Ptr{clGetDeviceImageInfoQCOM_t}

@checked function clGetDeviceImageInfoQCOM(device, image_width, image_height, image_format,
                                           param_name, param_value_size, param_value,
                                           param_value_size_ret)
    @ccall libopencl.clGetDeviceImageInfoQCOM(device::cl_device_id, image_width::Csize_t,
                                              image_height::Csize_t,
                                              image_format::Ptr{cl_image_format},
                                              param_name::cl_image_pitch_info_qcom,
                                              param_value_size::Csize_t,
                                              param_value::Ptr{Cvoid},
                                              param_value_size_ret::Ptr{Csize_t})::cl_int
end

struct _cl_mem_ion_host_ptr
    ext_host_ptr::cl_mem_ext_host_ptr
    ion_filedesc::Cint
    ion_hostptr::Ptr{Cvoid}
end

const cl_mem_ion_host_ptr = _cl_mem_ion_host_ptr

struct _cl_mem_android_native_buffer_host_ptr
    ext_host_ptr::cl_mem_ext_host_ptr
    anb_ptr::Ptr{Cvoid}
end

const cl_mem_android_native_buffer_host_ptr = _cl_mem_android_native_buffer_host_ptr

# typedef cl_int CL_API_CALL clEnqueueAcquireGrallocObjectsIMG_t ( cl_command_queue command_queue , cl_uint num_objects , const cl_mem * mem_objects , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueAcquireGrallocObjectsIMG_t = Cvoid

# typedef clEnqueueAcquireGrallocObjectsIMG_t * clEnqueueAcquireGrallocObjectsIMG_fn
const clEnqueueAcquireGrallocObjectsIMG_fn = Ptr{clEnqueueAcquireGrallocObjectsIMG_t}

# typedef cl_int CL_API_CALL clEnqueueReleaseGrallocObjectsIMG_t ( cl_command_queue command_queue , cl_uint num_objects , const cl_mem * mem_objects , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueReleaseGrallocObjectsIMG_t = Cvoid

# typedef clEnqueueReleaseGrallocObjectsIMG_t * clEnqueueReleaseGrallocObjectsIMG_fn
const clEnqueueReleaseGrallocObjectsIMG_fn = Ptr{clEnqueueReleaseGrallocObjectsIMG_t}

@checked function clEnqueueAcquireGrallocObjectsIMG(command_queue, num_objects, mem_objects,
                                                    num_events_in_wait_list,
                                                    event_wait_list, event)
    @ccall libopencl.clEnqueueAcquireGrallocObjectsIMG(command_queue::cl_command_queue,
                                                       num_objects::cl_uint,
                                                       mem_objects::Ptr{cl_mem},
                                                       num_events_in_wait_list::cl_uint,
                                                       event_wait_list::Ptr{cl_event},
                                                       event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueReleaseGrallocObjectsIMG(command_queue, num_objects, mem_objects,
                                                    num_events_in_wait_list,
                                                    event_wait_list, event)
    @ccall libopencl.clEnqueueReleaseGrallocObjectsIMG(command_queue::cl_command_queue,
                                                       num_objects::cl_uint,
                                                       mem_objects::Ptr{cl_mem},
                                                       num_events_in_wait_list::cl_uint,
                                                       event_wait_list::Ptr{cl_event},
                                                       event::Ptr{cl_event})::cl_int
end

const cl_mipmap_filter_mode_img = cl_uint

# typedef cl_int CL_API_CALL clEnqueueGenerateMipmapIMG_t ( cl_command_queue command_queue , cl_mem src_image , cl_mem dst_image , cl_mipmap_filter_mode_img mipmap_filter_mode , const size_t * array_region , const size_t * mip_region , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueGenerateMipmapIMG_t = Cvoid

# typedef clEnqueueGenerateMipmapIMG_t * clEnqueueGenerateMipmapIMG_fn
const clEnqueueGenerateMipmapIMG_fn = Ptr{clEnqueueGenerateMipmapIMG_t}

@checked function clEnqueueGenerateMipmapIMG(command_queue, src_image, dst_image,
                                             mipmap_filter_mode, array_region, mip_region,
                                             num_events_in_wait_list, event_wait_list,
                                             event)
    @ccall libopencl.clEnqueueGenerateMipmapIMG(command_queue::cl_command_queue,
                                                src_image::cl_mem, dst_image::cl_mem,
                                                mipmap_filter_mode::cl_mipmap_filter_mode_img,
                                                array_region::Ptr{Csize_t},
                                                mip_region::Ptr{Csize_t},
                                                num_events_in_wait_list::cl_uint,
                                                event_wait_list::Ptr{cl_event},
                                                event::Ptr{cl_event})::cl_int
end

# typedef cl_int CL_API_CALL clGetKernelSubGroupInfoKHR_t ( cl_kernel in_kernel , cl_device_id in_device , cl_kernel_sub_group_info param_name , size_t input_value_size , const void * input_value , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetKernelSubGroupInfoKHR_t = Cvoid

# typedef clGetKernelSubGroupInfoKHR_t * clGetKernelSubGroupInfoKHR_fn
const clGetKernelSubGroupInfoKHR_fn = Ptr{clGetKernelSubGroupInfoKHR_t}

@checked function clGetKernelSubGroupInfoKHR(in_kernel, in_device, param_name,
                                             input_value_size, input_value,
                                             param_value_size, param_value,
                                             param_value_size_ret)
    @ccall libopencl.clGetKernelSubGroupInfoKHR(in_kernel::cl_kernel,
                                                in_device::cl_device_id,
                                                param_name::cl_kernel_sub_group_info,
                                                input_value_size::Csize_t,
                                                input_value::Ptr{Cvoid},
                                                param_value_size::Csize_t,
                                                param_value::Ptr{Cvoid},
                                                param_value_size_ret::Ptr{Csize_t})::cl_int
end

const cl_queue_priority_khr = cl_uint

const cl_queue_throttle_khr = cl_uint

const cl_version_khr = cl_uint

struct _cl_name_version_khr
    version::cl_version_khr
    name::NTuple{64,Cchar}
end

const cl_name_version_khr = _cl_name_version_khr

struct _cl_device_pci_bus_info_khr
    pci_domain::cl_uint
    pci_bus::cl_uint
    pci_device::cl_uint
    pci_function::cl_uint
end

const cl_device_pci_bus_info_khr = _cl_device_pci_bus_info_khr

# typedef cl_int CL_API_CALL clGetKernelSuggestedLocalWorkSizeKHR_t ( cl_command_queue command_queue , cl_kernel kernel , cl_uint work_dim , const size_t * global_work_offset , const size_t * global_work_size , size_t * suggested_local_work_size )
const clGetKernelSuggestedLocalWorkSizeKHR_t = Cvoid

# typedef clGetKernelSuggestedLocalWorkSizeKHR_t * clGetKernelSuggestedLocalWorkSizeKHR_fn
const clGetKernelSuggestedLocalWorkSizeKHR_fn = Ptr{clGetKernelSuggestedLocalWorkSizeKHR_t}

@checked function clGetKernelSuggestedLocalWorkSizeKHR(command_queue, kernel, work_dim,
                                                       global_work_offset, global_work_size,
                                                       suggested_local_work_size)
    @ccall libopencl.clGetKernelSuggestedLocalWorkSizeKHR(command_queue::cl_command_queue,
                                                          kernel::cl_kernel,
                                                          work_dim::cl_uint,
                                                          global_work_offset::Ptr{Csize_t},
                                                          global_work_size::Ptr{Csize_t},
                                                          suggested_local_work_size::Ptr{Csize_t})::cl_int
end

const cl_device_integer_dot_product_capabilities_khr = cl_bitfield

struct _cl_device_integer_dot_product_acceleration_properties_khr
    signed_accelerated::cl_bool
    unsigned_accelerated::cl_bool
    mixed_signedness_accelerated::cl_bool
    accumulating_saturating_signed_accelerated::cl_bool
    accumulating_saturating_unsigned_accelerated::cl_bool
    accumulating_saturating_mixed_signedness_accelerated::cl_bool
end

const cl_device_integer_dot_product_acceleration_properties_khr = _cl_device_integer_dot_product_acceleration_properties_khr

const cl_external_memory_handle_type_khr = cl_uint

# typedef cl_int CL_API_CALL clEnqueueAcquireExternalMemObjectsKHR_t ( cl_command_queue command_queue , cl_uint num_mem_objects , const cl_mem * mem_objects , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueAcquireExternalMemObjectsKHR_t = Cvoid

# typedef clEnqueueAcquireExternalMemObjectsKHR_t * clEnqueueAcquireExternalMemObjectsKHR_fn
const clEnqueueAcquireExternalMemObjectsKHR_fn = Ptr{clEnqueueAcquireExternalMemObjectsKHR_t}

# typedef cl_int CL_API_CALL clEnqueueReleaseExternalMemObjectsKHR_t ( cl_command_queue command_queue , cl_uint num_mem_objects , const cl_mem * mem_objects , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueReleaseExternalMemObjectsKHR_t = Cvoid

# typedef clEnqueueReleaseExternalMemObjectsKHR_t * clEnqueueReleaseExternalMemObjectsKHR_fn
const clEnqueueReleaseExternalMemObjectsKHR_fn = Ptr{clEnqueueReleaseExternalMemObjectsKHR_t}

@checked function clEnqueueAcquireExternalMemObjectsKHR(command_queue, num_mem_objects,
                                                        mem_objects,
                                                        num_events_in_wait_list,
                                                        event_wait_list, event)
    @ccall libopencl.clEnqueueAcquireExternalMemObjectsKHR(command_queue::cl_command_queue,
                                                           num_mem_objects::cl_uint,
                                                           mem_objects::Ptr{cl_mem},
                                                           num_events_in_wait_list::cl_uint,
                                                           event_wait_list::Ptr{cl_event},
                                                           event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueReleaseExternalMemObjectsKHR(command_queue, num_mem_objects,
                                                        mem_objects,
                                                        num_events_in_wait_list,
                                                        event_wait_list, event)
    @ccall libopencl.clEnqueueReleaseExternalMemObjectsKHR(command_queue::cl_command_queue,
                                                           num_mem_objects::cl_uint,
                                                           mem_objects::Ptr{cl_mem},
                                                           num_events_in_wait_list::cl_uint,
                                                           event_wait_list::Ptr{cl_event},
                                                           event::Ptr{cl_event})::cl_int
end

mutable struct _cl_semaphore_khr end

const cl_semaphore_khr = Ptr{_cl_semaphore_khr}

const cl_external_semaphore_handle_type_khr = cl_uint

# typedef cl_int CL_API_CALL clGetSemaphoreHandleForTypeKHR_t ( cl_semaphore_khr sema_object , cl_device_id device , cl_external_semaphore_handle_type_khr handle_type , size_t handle_size , void * handle_ptr , size_t * handle_size_ret )
const clGetSemaphoreHandleForTypeKHR_t = Cvoid

# typedef clGetSemaphoreHandleForTypeKHR_t * clGetSemaphoreHandleForTypeKHR_fn
const clGetSemaphoreHandleForTypeKHR_fn = Ptr{clGetSemaphoreHandleForTypeKHR_t}

@checked function clGetSemaphoreHandleForTypeKHR(sema_object, device, handle_type,
                                                 handle_size, handle_ptr, handle_size_ret)
    @ccall libopencl.clGetSemaphoreHandleForTypeKHR(sema_object::cl_semaphore_khr,
                                                    device::cl_device_id,
                                                    handle_type::cl_external_semaphore_handle_type_khr,
                                                    handle_size::Csize_t,
                                                    handle_ptr::Ptr{Cvoid},
                                                    handle_size_ret::Ptr{Csize_t})::cl_int
end

const cl_semaphore_reimport_properties_khr = cl_properties

# typedef cl_int CL_API_CALL clReImportSemaphoreSyncFdKHR_t ( cl_semaphore_khr sema_object , cl_semaphore_reimport_properties_khr * reimport_props , int fd )
const clReImportSemaphoreSyncFdKHR_t = Cvoid

# typedef clReImportSemaphoreSyncFdKHR_t * clReImportSemaphoreSyncFdKHR_fn
const clReImportSemaphoreSyncFdKHR_fn = Ptr{clReImportSemaphoreSyncFdKHR_t}

@checked function clReImportSemaphoreSyncFdKHR(sema_object, reimport_props, fd)
    @ccall libopencl.clReImportSemaphoreSyncFdKHR(sema_object::cl_semaphore_khr,
                                                  reimport_props::Ptr{cl_semaphore_reimport_properties_khr},
                                                  fd::Cint)::cl_int
end

const cl_semaphore_properties_khr = cl_properties

const cl_semaphore_info_khr = cl_uint

const cl_semaphore_type_khr = cl_uint

const cl_semaphore_payload_khr = cl_ulong

# typedef cl_semaphore_khr CL_API_CALL clCreateSemaphoreWithPropertiesKHR_t ( cl_context context , const cl_semaphore_properties_khr * sema_props , cl_int * errcode_ret )
const clCreateSemaphoreWithPropertiesKHR_t = Cvoid

# typedef clCreateSemaphoreWithPropertiesKHR_t * clCreateSemaphoreWithPropertiesKHR_fn
const clCreateSemaphoreWithPropertiesKHR_fn = Ptr{clCreateSemaphoreWithPropertiesKHR_t}

# typedef cl_int CL_API_CALL clEnqueueWaitSemaphoresKHR_t ( cl_command_queue command_queue , cl_uint num_sema_objects , const cl_semaphore_khr * sema_objects , const cl_semaphore_payload_khr * sema_payload_list , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueWaitSemaphoresKHR_t = Cvoid

# typedef clEnqueueWaitSemaphoresKHR_t * clEnqueueWaitSemaphoresKHR_fn
const clEnqueueWaitSemaphoresKHR_fn = Ptr{clEnqueueWaitSemaphoresKHR_t}

# typedef cl_int CL_API_CALL clEnqueueSignalSemaphoresKHR_t ( cl_command_queue command_queue , cl_uint num_sema_objects , const cl_semaphore_khr * sema_objects , const cl_semaphore_payload_khr * sema_payload_list , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueSignalSemaphoresKHR_t = Cvoid

# typedef clEnqueueSignalSemaphoresKHR_t * clEnqueueSignalSemaphoresKHR_fn
const clEnqueueSignalSemaphoresKHR_fn = Ptr{clEnqueueSignalSemaphoresKHR_t}

# typedef cl_int CL_API_CALL clGetSemaphoreInfoKHR_t ( cl_semaphore_khr sema_object , cl_semaphore_info_khr param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetSemaphoreInfoKHR_t = Cvoid

# typedef clGetSemaphoreInfoKHR_t * clGetSemaphoreInfoKHR_fn
const clGetSemaphoreInfoKHR_fn = Ptr{clGetSemaphoreInfoKHR_t}

# typedef cl_int CL_API_CALL clReleaseSemaphoreKHR_t ( cl_semaphore_khr sema_object )
const clReleaseSemaphoreKHR_t = Cvoid

# typedef clReleaseSemaphoreKHR_t * clReleaseSemaphoreKHR_fn
const clReleaseSemaphoreKHR_fn = Ptr{clReleaseSemaphoreKHR_t}

# typedef cl_int CL_API_CALL clRetainSemaphoreKHR_t ( cl_semaphore_khr sema_object )
const clRetainSemaphoreKHR_t = Cvoid

# typedef clRetainSemaphoreKHR_t * clRetainSemaphoreKHR_fn
const clRetainSemaphoreKHR_fn = Ptr{clRetainSemaphoreKHR_t}

function clCreateSemaphoreWithPropertiesKHR(context, sema_props, errcode_ret)
    @ccall libopencl.clCreateSemaphoreWithPropertiesKHR(context::cl_context,
                                                        sema_props::Ptr{cl_semaphore_properties_khr},
                                                        errcode_ret::Ptr{cl_int})::cl_semaphore_khr
end

@checked function clEnqueueWaitSemaphoresKHR(command_queue, num_sema_objects, sema_objects,
                                             sema_payload_list, num_events_in_wait_list,
                                             event_wait_list, event)
    @ccall libopencl.clEnqueueWaitSemaphoresKHR(command_queue::cl_command_queue,
                                                num_sema_objects::cl_uint,
                                                sema_objects::Ptr{cl_semaphore_khr},
                                                sema_payload_list::Ptr{cl_semaphore_payload_khr},
                                                num_events_in_wait_list::cl_uint,
                                                event_wait_list::Ptr{cl_event},
                                                event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSignalSemaphoresKHR(command_queue, num_sema_objects,
                                               sema_objects, sema_payload_list,
                                               num_events_in_wait_list, event_wait_list,
                                               event)
    @ccall libopencl.clEnqueueSignalSemaphoresKHR(command_queue::cl_command_queue,
                                                  num_sema_objects::cl_uint,
                                                  sema_objects::Ptr{cl_semaphore_khr},
                                                  sema_payload_list::Ptr{cl_semaphore_payload_khr},
                                                  num_events_in_wait_list::cl_uint,
                                                  event_wait_list::Ptr{cl_event},
                                                  event::Ptr{cl_event})::cl_int
end

@checked function clGetSemaphoreInfoKHR(sema_object, param_name, param_value_size,
                                        param_value, param_value_size_ret)
    @ccall libopencl.clGetSemaphoreInfoKHR(sema_object::cl_semaphore_khr,
                                           param_name::cl_semaphore_info_khr,
                                           param_value_size::Csize_t,
                                           param_value::Ptr{Cvoid},
                                           param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clReleaseSemaphoreKHR(sema_object)
    @ccall libopencl.clReleaseSemaphoreKHR(sema_object::cl_semaphore_khr)::cl_int
end

@checked function clRetainSemaphoreKHR(sema_object)
    @ccall libopencl.clRetainSemaphoreKHR(sema_object::cl_semaphore_khr)::cl_int
end

const cl_import_properties_arm = intptr_t

# typedef cl_mem CL_API_CALL clImportMemoryARM_t ( cl_context context , cl_mem_flags flags , const cl_import_properties_arm * properties , void * memory , size_t size , cl_int * errcode_ret )
const clImportMemoryARM_t = Cvoid

# typedef clImportMemoryARM_t * clImportMemoryARM_fn
const clImportMemoryARM_fn = Ptr{clImportMemoryARM_t}

function clImportMemoryARM(context, flags, properties, memory, size, errcode_ret)
    @ccall libopencl.clImportMemoryARM(context::cl_context, flags::cl_mem_flags,
                                       properties::Ptr{cl_import_properties_arm},
                                       memory::Ptr{Cvoid}, size::Csize_t,
                                       errcode_ret::Ptr{cl_int})::cl_mem
end

const cl_svm_mem_flags_arm = cl_bitfield

const cl_kernel_exec_info_arm = cl_uint

const cl_device_svm_capabilities_arm = cl_bitfield

# typedef void * CL_API_CALL clSVMAllocARM_t ( cl_context context , cl_svm_mem_flags_arm flags , size_t size , cl_uint alignment )
const clSVMAllocARM_t = Cvoid

# typedef clSVMAllocARM_t * clSVMAllocARM_fn
const clSVMAllocARM_fn = Ptr{clSVMAllocARM_t}

# typedef void CL_API_CALL clSVMFreeARM_t ( cl_context context , void * svm_pointer )
const clSVMFreeARM_t = Cvoid

# typedef clSVMFreeARM_t * clSVMFreeARM_fn
const clSVMFreeARM_fn = Ptr{clSVMFreeARM_t}

# typedef cl_int CL_API_CALL clEnqueueSVMFreeARM_t ( cl_command_queue command_queue , cl_uint num_svm_pointers , void * svm_pointers [ ] , void ( CL_CALLBACK * pfn_free_func ) ( cl_command_queue queue , cl_uint num_svm_pointers , void * svm_pointers [ ] , void * user_data ) , void * user_data , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueSVMFreeARM_t = Cvoid

# typedef clEnqueueSVMFreeARM_t * clEnqueueSVMFreeARM_fn
const clEnqueueSVMFreeARM_fn = Ptr{clEnqueueSVMFreeARM_t}

# typedef cl_int CL_API_CALL clEnqueueSVMMemcpyARM_t ( cl_command_queue command_queue , cl_bool blocking_copy , void * dst_ptr , const void * src_ptr , size_t size , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueSVMMemcpyARM_t = Cvoid

# typedef clEnqueueSVMMemcpyARM_t * clEnqueueSVMMemcpyARM_fn
const clEnqueueSVMMemcpyARM_fn = Ptr{clEnqueueSVMMemcpyARM_t}

# typedef cl_int CL_API_CALL clEnqueueSVMMemFillARM_t ( cl_command_queue command_queue , void * svm_ptr , const void * pattern , size_t pattern_size , size_t size , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueSVMMemFillARM_t = Cvoid

# typedef clEnqueueSVMMemFillARM_t * clEnqueueSVMMemFillARM_fn
const clEnqueueSVMMemFillARM_fn = Ptr{clEnqueueSVMMemFillARM_t}

# typedef cl_int CL_API_CALL clEnqueueSVMMapARM_t ( cl_command_queue command_queue , cl_bool blocking_map , cl_map_flags flags , void * svm_ptr , size_t size , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueSVMMapARM_t = Cvoid

# typedef clEnqueueSVMMapARM_t * clEnqueueSVMMapARM_fn
const clEnqueueSVMMapARM_fn = Ptr{clEnqueueSVMMapARM_t}

# typedef cl_int CL_API_CALL clEnqueueSVMUnmapARM_t ( cl_command_queue command_queue , void * svm_ptr , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueSVMUnmapARM_t = Cvoid

# typedef clEnqueueSVMUnmapARM_t * clEnqueueSVMUnmapARM_fn
const clEnqueueSVMUnmapARM_fn = Ptr{clEnqueueSVMUnmapARM_t}

# typedef cl_int CL_API_CALL clSetKernelArgSVMPointerARM_t ( cl_kernel kernel , cl_uint arg_index , const void * arg_value )
const clSetKernelArgSVMPointerARM_t = Cvoid

# typedef clSetKernelArgSVMPointerARM_t * clSetKernelArgSVMPointerARM_fn
const clSetKernelArgSVMPointerARM_fn = Ptr{clSetKernelArgSVMPointerARM_t}

# typedef cl_int CL_API_CALL clSetKernelExecInfoARM_t ( cl_kernel kernel , cl_kernel_exec_info_arm param_name , size_t param_value_size , const void * param_value )
const clSetKernelExecInfoARM_t = Cvoid

# typedef clSetKernelExecInfoARM_t * clSetKernelExecInfoARM_fn
const clSetKernelExecInfoARM_fn = Ptr{clSetKernelExecInfoARM_t}

function clSVMAllocARM(context, flags, size, alignment)
    @ccall libopencl.clSVMAllocARM(context::cl_context, flags::cl_svm_mem_flags_arm,
                                   size::Csize_t, alignment::cl_uint)::Ptr{Cvoid}
end

function clSVMFreeARM(context, svm_pointer)
    @ccall libopencl.clSVMFreeARM(context::cl_context, svm_pointer::Ptr{Cvoid})::Cvoid
end

@checked function clEnqueueSVMFreeARM(command_queue, num_svm_pointers, svm_pointers,
                                      pfn_free_func, user_data, num_events_in_wait_list,
                                      event_wait_list, event)
    @ccall libopencl.clEnqueueSVMFreeARM(command_queue::cl_command_queue,
                                         num_svm_pointers::cl_uint,
                                         svm_pointers::Ptr{Ptr{Cvoid}},
                                         pfn_free_func::Ptr{Cvoid}, user_data::Ptr{Cvoid},
                                         num_events_in_wait_list::cl_uint,
                                         event_wait_list::Ptr{cl_event},
                                         event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMMemcpyARM(command_queue, blocking_copy, dst_ptr, src_ptr,
                                        size, num_events_in_wait_list, event_wait_list,
                                        event)
    @ccall libopencl.clEnqueueSVMMemcpyARM(command_queue::cl_command_queue,
                                           blocking_copy::cl_bool, dst_ptr::Ptr{Cvoid},
                                           src_ptr::Ptr{Cvoid}, size::Csize_t,
                                           num_events_in_wait_list::cl_uint,
                                           event_wait_list::Ptr{cl_event},
                                           event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMMemFillARM(command_queue, svm_ptr, pattern, pattern_size,
                                         size, num_events_in_wait_list, event_wait_list,
                                         event)
    @ccall libopencl.clEnqueueSVMMemFillARM(command_queue::cl_command_queue,
                                            svm_ptr::Ptr{Cvoid}, pattern::Ptr{Cvoid},
                                            pattern_size::Csize_t, size::Csize_t,
                                            num_events_in_wait_list::cl_uint,
                                            event_wait_list::Ptr{cl_event},
                                            event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMMapARM(command_queue, blocking_map, flags, svm_ptr, size,
                                     num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueSVMMapARM(command_queue::cl_command_queue,
                                        blocking_map::cl_bool, flags::cl_map_flags,
                                        svm_ptr::Ptr{Cvoid}, size::Csize_t,
                                        num_events_in_wait_list::cl_uint,
                                        event_wait_list::Ptr{cl_event},
                                        event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueSVMUnmapARM(command_queue, svm_ptr, num_events_in_wait_list,
                                       event_wait_list, event)
    @ccall libopencl.clEnqueueSVMUnmapARM(command_queue::cl_command_queue,
                                          svm_ptr::Ptr{Cvoid},
                                          num_events_in_wait_list::cl_uint,
                                          event_wait_list::Ptr{cl_event},
                                          event::Ptr{cl_event})::cl_int
end

@checked function clSetKernelArgSVMPointerARM(kernel, arg_index, arg_value)
    @ccall libopencl.clSetKernelArgSVMPointerARM(kernel::cl_kernel, arg_index::cl_uint,
                                                 arg_value::Ptr{Cvoid})::cl_int
end

@checked function clSetKernelExecInfoARM(kernel, param_name, param_value_size, param_value)
    @ccall libopencl.clSetKernelExecInfoARM(kernel::cl_kernel,
                                            param_name::cl_kernel_exec_info_arm,
                                            param_value_size::Csize_t,
                                            param_value::Ptr{Cvoid})::cl_int
end

const cl_device_scheduling_controls_capabilities_arm = cl_bitfield

const cl_device_controlled_termination_capabilities_arm = cl_bitfield

const cl_device_feature_capabilities_intel = cl_bitfield

mutable struct _cl_accelerator_intel end

const cl_accelerator_intel = Ptr{_cl_accelerator_intel}

const cl_accelerator_type_intel = cl_uint

const cl_accelerator_info_intel = cl_uint

# typedef cl_accelerator_intel CL_API_CALL clCreateAcceleratorINTEL_t ( cl_context context , cl_accelerator_type_intel accelerator_type , size_t descriptor_size , const void * descriptor , cl_int * errcode_ret )
const clCreateAcceleratorINTEL_t = Cvoid

# typedef clCreateAcceleratorINTEL_t * clCreateAcceleratorINTEL_fn
const clCreateAcceleratorINTEL_fn = Ptr{clCreateAcceleratorINTEL_t}

# typedef cl_int CL_API_CALL clGetAcceleratorInfoINTEL_t ( cl_accelerator_intel accelerator , cl_accelerator_info_intel param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetAcceleratorInfoINTEL_t = Cvoid

# typedef clGetAcceleratorInfoINTEL_t * clGetAcceleratorInfoINTEL_fn
const clGetAcceleratorInfoINTEL_fn = Ptr{clGetAcceleratorInfoINTEL_t}

# typedef cl_int CL_API_CALL clRetainAcceleratorINTEL_t ( cl_accelerator_intel accelerator )
const clRetainAcceleratorINTEL_t = Cvoid

# typedef clRetainAcceleratorINTEL_t * clRetainAcceleratorINTEL_fn
const clRetainAcceleratorINTEL_fn = Ptr{clRetainAcceleratorINTEL_t}

# typedef cl_int CL_API_CALL clReleaseAcceleratorINTEL_t ( cl_accelerator_intel accelerator )
const clReleaseAcceleratorINTEL_t = Cvoid

# typedef clReleaseAcceleratorINTEL_t * clReleaseAcceleratorINTEL_fn
const clReleaseAcceleratorINTEL_fn = Ptr{clReleaseAcceleratorINTEL_t}

function clCreateAcceleratorINTEL(context, accelerator_type, descriptor_size, descriptor,
                                  errcode_ret)
    @ccall libopencl.clCreateAcceleratorINTEL(context::cl_context,
                                              accelerator_type::cl_accelerator_type_intel,
                                              descriptor_size::Csize_t,
                                              descriptor::Ptr{Cvoid},
                                              errcode_ret::Ptr{cl_int})::cl_accelerator_intel
end

@checked function clGetAcceleratorInfoINTEL(accelerator, param_name, param_value_size,
                                            param_value, param_value_size_ret)
    @ccall libopencl.clGetAcceleratorInfoINTEL(accelerator::cl_accelerator_intel,
                                               param_name::cl_accelerator_info_intel,
                                               param_value_size::Csize_t,
                                               param_value::Ptr{Cvoid},
                                               param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clRetainAcceleratorINTEL(accelerator)
    @ccall libopencl.clRetainAcceleratorINTEL(accelerator::cl_accelerator_intel)::cl_int
end

@checked function clReleaseAcceleratorINTEL(accelerator)
    @ccall libopencl.clReleaseAcceleratorINTEL(accelerator::cl_accelerator_intel)::cl_int
end

struct _cl_motion_estimation_desc_intel
    mb_block_type::cl_uint
    subpixel_mode::cl_uint
    sad_adjust_mode::cl_uint
    search_path_type::cl_uint
end

const cl_motion_estimation_desc_intel = _cl_motion_estimation_desc_intel

const cl_diagnostics_verbose_level = cl_uint

const cl_device_unified_shared_memory_capabilities_intel = cl_bitfield

const cl_mem_properties_intel = cl_properties

const cl_mem_alloc_flags_intel = cl_bitfield

const cl_mem_info_intel = cl_uint

const cl_unified_shared_memory_type_intel = cl_uint

const cl_mem_advice_intel = cl_uint

# typedef void * CL_API_CALL clHostMemAllocINTEL_t ( cl_context context , const cl_mem_properties_intel * properties , size_t size , cl_uint alignment , cl_int * errcode_ret )
const clHostMemAllocINTEL_t = Cvoid

# typedef clHostMemAllocINTEL_t * clHostMemAllocINTEL_fn
const clHostMemAllocINTEL_fn = Ptr{clHostMemAllocINTEL_t}

# typedef void * CL_API_CALL clDeviceMemAllocINTEL_t ( cl_context context , cl_device_id device , const cl_mem_properties_intel * properties , size_t size , cl_uint alignment , cl_int * errcode_ret )
const clDeviceMemAllocINTEL_t = Cvoid

# typedef clDeviceMemAllocINTEL_t * clDeviceMemAllocINTEL_fn
const clDeviceMemAllocINTEL_fn = Ptr{clDeviceMemAllocINTEL_t}

# typedef void * CL_API_CALL clSharedMemAllocINTEL_t ( cl_context context , cl_device_id device , const cl_mem_properties_intel * properties , size_t size , cl_uint alignment , cl_int * errcode_ret )
const clSharedMemAllocINTEL_t = Cvoid

# typedef clSharedMemAllocINTEL_t * clSharedMemAllocINTEL_fn
const clSharedMemAllocINTEL_fn = Ptr{clSharedMemAllocINTEL_t}

# typedef cl_int CL_API_CALL clMemFreeINTEL_t ( cl_context context , void * ptr )
const clMemFreeINTEL_t = Cvoid

# typedef clMemFreeINTEL_t * clMemFreeINTEL_fn
const clMemFreeINTEL_fn = Ptr{clMemFreeINTEL_t}

# typedef cl_int CL_API_CALL clMemBlockingFreeINTEL_t ( cl_context context , void * ptr )
const clMemBlockingFreeINTEL_t = Cvoid

# typedef clMemBlockingFreeINTEL_t * clMemBlockingFreeINTEL_fn
const clMemBlockingFreeINTEL_fn = Ptr{clMemBlockingFreeINTEL_t}

# typedef cl_int CL_API_CALL clGetMemAllocInfoINTEL_t ( cl_context context , const void * ptr , cl_mem_info_intel param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetMemAllocInfoINTEL_t = Cvoid

# typedef clGetMemAllocInfoINTEL_t * clGetMemAllocInfoINTEL_fn
const clGetMemAllocInfoINTEL_fn = Ptr{clGetMemAllocInfoINTEL_t}

# typedef cl_int CL_API_CALL clSetKernelArgMemPointerINTEL_t ( cl_kernel kernel , cl_uint arg_index , const void * arg_value )
const clSetKernelArgMemPointerINTEL_t = Cvoid

# typedef clSetKernelArgMemPointerINTEL_t * clSetKernelArgMemPointerINTEL_fn
const clSetKernelArgMemPointerINTEL_fn = Ptr{clSetKernelArgMemPointerINTEL_t}

# typedef cl_int CL_API_CALL clEnqueueMemFillINTEL_t ( cl_command_queue command_queue , void * dst_ptr , const void * pattern , size_t pattern_size , size_t size , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueMemFillINTEL_t = Cvoid

# typedef clEnqueueMemFillINTEL_t * clEnqueueMemFillINTEL_fn
const clEnqueueMemFillINTEL_fn = Ptr{clEnqueueMemFillINTEL_t}

# typedef cl_int CL_API_CALL clEnqueueMemcpyINTEL_t ( cl_command_queue command_queue , cl_bool blocking , void * dst_ptr , const void * src_ptr , size_t size , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueMemcpyINTEL_t = Cvoid

# typedef clEnqueueMemcpyINTEL_t * clEnqueueMemcpyINTEL_fn
const clEnqueueMemcpyINTEL_fn = Ptr{clEnqueueMemcpyINTEL_t}

# typedef cl_int CL_API_CALL clEnqueueMemAdviseINTEL_t ( cl_command_queue command_queue , const void * ptr , size_t size , cl_mem_advice_intel advice , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueMemAdviseINTEL_t = Cvoid

# typedef clEnqueueMemAdviseINTEL_t * clEnqueueMemAdviseINTEL_fn
const clEnqueueMemAdviseINTEL_fn = Ptr{clEnqueueMemAdviseINTEL_t}

function clHostMemAllocINTEL(context, properties, size, alignment, errcode_ret)
    @ccall libopencl.clHostMemAllocINTEL(context::cl_context,
                                         properties::Ptr{cl_mem_properties_intel},
                                         size::Csize_t, alignment::cl_uint,
                                         errcode_ret::Ptr{cl_int})::Ptr{Cvoid}
end

function clDeviceMemAllocINTEL(context, device, properties, size, alignment, errcode_ret)
    @ccall libopencl.clDeviceMemAllocINTEL(context::cl_context, device::cl_device_id,
                                           properties::Ptr{cl_mem_properties_intel},
                                           size::Csize_t, alignment::cl_uint,
                                           errcode_ret::Ptr{cl_int})::Ptr{Cvoid}
end

function clSharedMemAllocINTEL(context, device, properties, size, alignment, errcode_ret)
    @ccall libopencl.clSharedMemAllocINTEL(context::cl_context, device::cl_device_id,
                                           properties::Ptr{cl_mem_properties_intel},
                                           size::Csize_t, alignment::cl_uint,
                                           errcode_ret::Ptr{cl_int})::Ptr{Cvoid}
end

@checked function clMemFreeINTEL(context, ptr)
    @ccall libopencl.clMemFreeINTEL(context::cl_context, ptr::Ptr{Cvoid})::cl_int
end

@checked function clMemBlockingFreeINTEL(context, ptr)
    @ccall libopencl.clMemBlockingFreeINTEL(context::cl_context, ptr::Ptr{Cvoid})::cl_int
end

@checked function clGetMemAllocInfoINTEL(context, ptr, param_name, param_value_size,
                                         param_value, param_value_size_ret)
    @ccall libopencl.clGetMemAllocInfoINTEL(context::cl_context, ptr::Ptr{Cvoid},
                                            param_name::cl_mem_info_intel,
                                            param_value_size::Csize_t,
                                            param_value::Ptr{Cvoid},
                                            param_value_size_ret::Ptr{Csize_t})::cl_int
end

@checked function clSetKernelArgMemPointerINTEL(kernel, arg_index, arg_value)
    @ccall libopencl.clSetKernelArgMemPointerINTEL(kernel::cl_kernel, arg_index::cl_uint,
                                                   arg_value::Ptr{Cvoid})::cl_int
end

@checked function clEnqueueMemFillINTEL(command_queue, dst_ptr, pattern, pattern_size, size,
                                        num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueMemFillINTEL(command_queue::cl_command_queue,
                                           dst_ptr::Ptr{Cvoid}, pattern::Ptr{Cvoid},
                                           pattern_size::Csize_t, size::Csize_t,
                                           num_events_in_wait_list::cl_uint,
                                           event_wait_list::Ptr{cl_event},
                                           event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueMemcpyINTEL(command_queue, blocking, dst_ptr, src_ptr, size,
                                       num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueMemcpyINTEL(command_queue::cl_command_queue,
                                          blocking::cl_bool, dst_ptr::Ptr{Cvoid},
                                          src_ptr::Ptr{Cvoid}, size::Csize_t,
                                          num_events_in_wait_list::cl_uint,
                                          event_wait_list::Ptr{cl_event},
                                          event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueMemAdviseINTEL(command_queue, ptr, size, advice,
                                          num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueMemAdviseINTEL(command_queue::cl_command_queue,
                                             ptr::Ptr{Cvoid}, size::Csize_t,
                                             advice::cl_mem_advice_intel,
                                             num_events_in_wait_list::cl_uint,
                                             event_wait_list::Ptr{cl_event},
                                             event::Ptr{cl_event})::cl_int
end

# typedef cl_int CL_API_CALL clEnqueueMigrateMemINTEL_t ( cl_command_queue command_queue , const void * ptr , size_t size , cl_mem_migration_flags flags , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueMigrateMemINTEL_t = Cvoid

# typedef clEnqueueMigrateMemINTEL_t * clEnqueueMigrateMemINTEL_fn
const clEnqueueMigrateMemINTEL_fn = Ptr{clEnqueueMigrateMemINTEL_t}

@checked function clEnqueueMigrateMemINTEL(command_queue, ptr, size, flags,
                                           num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueMigrateMemINTEL(command_queue::cl_command_queue,
                                              ptr::Ptr{Cvoid}, size::Csize_t,
                                              flags::cl_mem_migration_flags,
                                              num_events_in_wait_list::cl_uint,
                                              event_wait_list::Ptr{cl_event},
                                              event::Ptr{cl_event})::cl_int
end

# typedef cl_int CL_API_CALL clEnqueueMemsetINTEL_t ( cl_command_queue command_queue , void * dst_ptr , cl_int value , size_t size , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueMemsetINTEL_t = Cvoid

# typedef clEnqueueMemsetINTEL_t * clEnqueueMemsetINTEL_fn
const clEnqueueMemsetINTEL_fn = Ptr{clEnqueueMemsetINTEL_t}

@checked function clEnqueueMemsetINTEL(command_queue, dst_ptr, value, size,
                                       num_events_in_wait_list, event_wait_list, event)
    @ccall libopencl.clEnqueueMemsetINTEL(command_queue::cl_command_queue,
                                          dst_ptr::Ptr{Cvoid}, value::cl_int, size::Csize_t,
                                          num_events_in_wait_list::cl_uint,
                                          event_wait_list::Ptr{cl_event},
                                          event::Ptr{cl_event})::cl_int
end

# typedef cl_mem CL_API_CALL clCreateBufferWithPropertiesINTEL_t ( cl_context context , const cl_mem_properties_intel * properties , cl_mem_flags flags , size_t size , void * host_ptr , cl_int * errcode_ret )
const clCreateBufferWithPropertiesINTEL_t = Cvoid

# typedef clCreateBufferWithPropertiesINTEL_t * clCreateBufferWithPropertiesINTEL_fn
const clCreateBufferWithPropertiesINTEL_fn = Ptr{clCreateBufferWithPropertiesINTEL_t}

function clCreateBufferWithPropertiesINTEL(context, properties, flags, size, host_ptr,
                                           errcode_ret)
    @ccall libopencl.clCreateBufferWithPropertiesINTEL(context::cl_context,
                                                       properties::Ptr{cl_mem_properties_intel},
                                                       flags::cl_mem_flags, size::Csize_t,
                                                       host_ptr::Ptr{Cvoid},
                                                       errcode_ret::Ptr{cl_int})::cl_mem
end

# typedef cl_int CL_API_CALL clEnqueueReadHostPipeINTEL_t ( cl_command_queue command_queue , cl_program program , const char * pipe_symbol , cl_bool blocking_read , void * ptr , size_t size , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueReadHostPipeINTEL_t = Cvoid

# typedef clEnqueueReadHostPipeINTEL_t * clEnqueueReadHostPipeINTEL_fn
const clEnqueueReadHostPipeINTEL_fn = Ptr{clEnqueueReadHostPipeINTEL_t}

# typedef cl_int CL_API_CALL clEnqueueWriteHostPipeINTEL_t ( cl_command_queue command_queue , cl_program program , const char * pipe_symbol , cl_bool blocking_write , const void * ptr , size_t size , cl_uint num_events_in_wait_list , const cl_event * event_wait_list , cl_event * event )
const clEnqueueWriteHostPipeINTEL_t = Cvoid

# typedef clEnqueueWriteHostPipeINTEL_t * clEnqueueWriteHostPipeINTEL_fn
const clEnqueueWriteHostPipeINTEL_fn = Ptr{clEnqueueWriteHostPipeINTEL_t}

@checked function clEnqueueReadHostPipeINTEL(command_queue, program, pipe_symbol,
                                             blocking_read, ptr, size,
                                             num_events_in_wait_list, event_wait_list,
                                             event)
    @ccall libopencl.clEnqueueReadHostPipeINTEL(command_queue::cl_command_queue,
                                                program::cl_program,
                                                pipe_symbol::Ptr{Cchar},
                                                blocking_read::cl_bool, ptr::Ptr{Cvoid},
                                                size::Csize_t,
                                                num_events_in_wait_list::cl_uint,
                                                event_wait_list::Ptr{cl_event},
                                                event::Ptr{cl_event})::cl_int
end

@checked function clEnqueueWriteHostPipeINTEL(command_queue, program, pipe_symbol,
                                              blocking_write, ptr, size,
                                              num_events_in_wait_list, event_wait_list,
                                              event)
    @ccall libopencl.clEnqueueWriteHostPipeINTEL(command_queue::cl_command_queue,
                                                 program::cl_program,
                                                 pipe_symbol::Ptr{Cchar},
                                                 blocking_write::cl_bool, ptr::Ptr{Cvoid},
                                                 size::Csize_t,
                                                 num_events_in_wait_list::cl_uint,
                                                 event_wait_list::Ptr{cl_event},
                                                 event::Ptr{cl_event})::cl_int
end

const cl_command_queue_capabilities_intel = cl_bitfield

struct _cl_queue_family_properties_intel
    properties::cl_command_queue_properties
    capabilities::cl_command_queue_capabilities_intel
    count::cl_uint
    name::NTuple{64,Cchar}
end

const cl_queue_family_properties_intel = _cl_queue_family_properties_intel

const cl_image_requirements_info_ext = cl_uint

# typedef cl_int CL_API_CALL clGetImageRequirementsInfoEXT_t ( cl_context context , const cl_mem_properties * properties , cl_mem_flags flags , const cl_image_format * image_format , const cl_image_desc * image_desc , cl_image_requirements_info_ext param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetImageRequirementsInfoEXT_t = Cvoid

# typedef clGetImageRequirementsInfoEXT_t * clGetImageRequirementsInfoEXT_fn
const clGetImageRequirementsInfoEXT_fn = Ptr{clGetImageRequirementsInfoEXT_t}

@checked function clGetImageRequirementsInfoEXT(context, properties, flags, image_format,
                                                image_desc, param_name, param_value_size,
                                                param_value, param_value_size_ret)
    @ccall libopencl.clGetImageRequirementsInfoEXT(context::cl_context,
                                                   properties::Ptr{cl_mem_properties},
                                                   flags::cl_mem_flags,
                                                   image_format::Ptr{cl_image_format},
                                                   image_desc::Ptr{cl_image_desc},
                                                   param_name::cl_image_requirements_info_ext,
                                                   param_value_size::Csize_t,
                                                   param_value::Ptr{Cvoid},
                                                   param_value_size_ret::Ptr{Csize_t})::cl_int
end

const cl_icdl_info = cl_uint

# typedef cl_int CL_API_CALL clGetICDLoaderInfoOCLICD_t ( cl_icdl_info param_name , size_t param_value_size , void * param_value , size_t * param_value_size_ret )
const clGetICDLoaderInfoOCLICD_t = Cvoid

# typedef clGetICDLoaderInfoOCLICD_t * clGetICDLoaderInfoOCLICD_fn
const clGetICDLoaderInfoOCLICD_fn = Ptr{clGetICDLoaderInfoOCLICD_t}

@checked function clGetICDLoaderInfoOCLICD(param_name, param_value_size, param_value,
                                           param_value_size_ret)
    @ccall libopencl.clGetICDLoaderInfoOCLICD(param_name::cl_icdl_info,
                                              param_value_size::Csize_t,
                                              param_value::Ptr{Cvoid},
                                              param_value_size_ret::Ptr{Csize_t})::cl_int
end

const cl_device_fp_atomic_capabilities_ext = cl_bitfield

# typedef cl_int CL_API_CALL clSetContentSizeBufferPoCL_t ( cl_mem buffer , cl_mem content_size_buffer )
const clSetContentSizeBufferPoCL_t = Cvoid

# typedef clSetContentSizeBufferPoCL_t * clSetContentSizeBufferPoCL_fn
const clSetContentSizeBufferPoCL_fn = Ptr{clSetContentSizeBufferPoCL_t}

@checked function clSetContentSizeBufferPoCL(buffer, content_size_buffer)
    @ccall libopencl.clSetContentSizeBufferPoCL(buffer::cl_mem,
                                                content_size_buffer::cl_mem)::cl_int
end

const cl_device_kernel_clock_capabilities_khr = cl_bitfield

# typedef cl_int CL_API_CALL clCancelCommandsIMG_t ( const cl_event * event_list , size_t num_events_in_list )
const clCancelCommandsIMG_t = Cvoid

# typedef clCancelCommandsIMG_t * clCancelCommandsIMG_fn
const clCancelCommandsIMG_fn = Ptr{clCancelCommandsIMG_t}

@checked function clCancelCommandsIMG(event_list, num_events_in_list)
    @ccall libopencl.clCancelCommandsIMG(event_list::Ptr{cl_event},
                                         num_events_in_list::Csize_t)::cl_int
end

const CL_NAME_VERSION_MAX_NAME_SIZE = 64

const CL_SUCCESS = 0

const CL_DEVICE_NOT_FOUND = -1

const CL_DEVICE_NOT_AVAILABLE = -2

const CL_COMPILER_NOT_AVAILABLE = -3

const CL_MEM_OBJECT_ALLOCATION_FAILURE = -4

const CL_OUT_OF_RESOURCES = -5

const CL_OUT_OF_HOST_MEMORY = -6

const CL_PROFILING_INFO_NOT_AVAILABLE = -7

const CL_MEM_COPY_OVERLAP = -8

const CL_IMAGE_FORMAT_MISMATCH = -9

const CL_IMAGE_FORMAT_NOT_SUPPORTED = -10

const CL_BUILD_PROGRAM_FAILURE = -11

const CL_MAP_FAILURE = -12

const CL_MISALIGNED_SUB_BUFFER_OFFSET = -13

const CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14

const CL_COMPILE_PROGRAM_FAILURE = -15

const CL_LINKER_NOT_AVAILABLE = -16

const CL_LINK_PROGRAM_FAILURE = -17

const CL_DEVICE_PARTITION_FAILED = -18

const CL_KERNEL_ARG_INFO_NOT_AVAILABLE = -19

const CL_INVALID_VALUE = -30

const CL_INVALID_DEVICE_TYPE = -31

const CL_INVALID_PLATFORM = -32

const CL_INVALID_DEVICE = -33

const CL_INVALID_CONTEXT = -34

const CL_INVALID_QUEUE_PROPERTIES = -35

const CL_INVALID_COMMAND_QUEUE = -36

const CL_INVALID_HOST_PTR = -37

const CL_INVALID_MEM_OBJECT = -38

const CL_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39

const CL_INVALID_IMAGE_SIZE = -40

const CL_INVALID_SAMPLER = -41

const CL_INVALID_BINARY = -42

const CL_INVALID_BUILD_OPTIONS = -43

const CL_INVALID_PROGRAM = -44

const CL_INVALID_PROGRAM_EXECUTABLE = -45

const CL_INVALID_KERNEL_NAME = -46

const CL_INVALID_KERNEL_DEFINITION = -47

const CL_INVALID_KERNEL = -48

const CL_INVALID_ARG_INDEX = -49

const CL_INVALID_ARG_VALUE = -50

const CL_INVALID_ARG_SIZE = -51

const CL_INVALID_KERNEL_ARGS = -52

const CL_INVALID_WORK_DIMENSION = -53

const CL_INVALID_WORK_GROUP_SIZE = -54

const CL_INVALID_WORK_ITEM_SIZE = -55

const CL_INVALID_GLOBAL_OFFSET = -56

const CL_INVALID_EVENT_WAIT_LIST = -57

const CL_INVALID_EVENT = -58

const CL_INVALID_OPERATION = -59

const CL_INVALID_GL_OBJECT = -60

const CL_INVALID_BUFFER_SIZE = -61

const CL_INVALID_MIP_LEVEL = -62

const CL_INVALID_GLOBAL_WORK_SIZE = -63

const CL_INVALID_PROPERTY = -64

const CL_INVALID_IMAGE_DESCRIPTOR = -65

const CL_INVALID_COMPILER_OPTIONS = -66

const CL_INVALID_LINKER_OPTIONS = -67

const CL_INVALID_DEVICE_PARTITION_COUNT = -68

const CL_INVALID_PIPE_SIZE = -69

const CL_INVALID_DEVICE_QUEUE = -70

const CL_INVALID_SPEC_ID = -71

const CL_MAX_SIZE_RESTRICTION_EXCEEDED = -72

const CL_FALSE = 0

const CL_TRUE = 1

const CL_BLOCKING = CL_TRUE

const CL_NON_BLOCKING = CL_FALSE

const CL_PLATFORM_PROFILE = 0x0900

const CL_PLATFORM_VERSION = 0x0901

const CL_PLATFORM_NAME = 0x0902

const CL_PLATFORM_VENDOR = 0x0903

const CL_PLATFORM_EXTENSIONS = 0x0904

const CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905

const CL_PLATFORM_NUMERIC_VERSION = 0x0906

const CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907

const CL_DEVICE_TYPE_DEFAULT = 1 << 0

const CL_DEVICE_TYPE_CPU = 1 << 1

const CL_DEVICE_TYPE_GPU = 1 << 2

const CL_DEVICE_TYPE_ACCELERATOR = 1 << 3

const CL_DEVICE_TYPE_CUSTOM = 1 << 4

const CL_DEVICE_TYPE_ALL = 0xffffffff

const CL_DEVICE_TYPE = 0x1000

const CL_DEVICE_VENDOR_ID = 0x1001

const CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002

const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003

const CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004

const CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100a

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100b

const CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100c

const CL_DEVICE_ADDRESS_BITS = 0x100d

const CL_DEVICE_MAX_READ_IMAGE_ARGS = 0x100e

const CL_DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100f

const CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010

const CL_DEVICE_IMAGE2D_MAX_WIDTH = 0x1011

const CL_DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012

const CL_DEVICE_IMAGE3D_MAX_WIDTH = 0x1013

const CL_DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014

const CL_DEVICE_IMAGE3D_MAX_DEPTH = 0x1015

const CL_DEVICE_IMAGE_SUPPORT = 0x1016

const CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017

const CL_DEVICE_MAX_SAMPLERS = 0x1018

const CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019

const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101a

const CL_DEVICE_SINGLE_FP_CONFIG = 0x101b

const CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101c

const CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101d

const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101e

const CL_DEVICE_GLOBAL_MEM_SIZE = 0x101f

const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020

const CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021

const CL_DEVICE_LOCAL_MEM_TYPE = 0x1022

const CL_DEVICE_LOCAL_MEM_SIZE = 0x1023

const CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024

const CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025

const CL_DEVICE_ENDIAN_LITTLE = 0x1026

const CL_DEVICE_AVAILABLE = 0x1027

const CL_DEVICE_COMPILER_AVAILABLE = 0x1028

const CL_DEVICE_EXECUTION_CAPABILITIES = 0x1029

const CL_DEVICE_QUEUE_PROPERTIES = 0x102a

const CL_DEVICE_QUEUE_ON_HOST_PROPERTIES = 0x102a

const CL_DEVICE_NAME = 0x102b

const CL_DEVICE_VENDOR = 0x102c

const CL_DRIVER_VERSION = 0x102d

const CL_DEVICE_PROFILE = 0x102e

const CL_DEVICE_VERSION = 0x102f

const CL_DEVICE_EXTENSIONS = 0x1030

const CL_DEVICE_PLATFORM = 0x1031

const CL_DEVICE_DOUBLE_FP_CONFIG = 0x1032

const CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034

const CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035

const CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036

const CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037

const CL_DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038

const CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039

const CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103a

const CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103b

const CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103c

const CL_DEVICE_OPENCL_C_VERSION = 0x103d

const CL_DEVICE_LINKER_AVAILABLE = 0x103e

const CL_DEVICE_BUILT_IN_KERNELS = 0x103f

const CL_DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040

const CL_DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041

const CL_DEVICE_PARENT_DEVICE = 0x1042

const CL_DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043

const CL_DEVICE_PARTITION_PROPERTIES = 0x1044

const CL_DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045

const CL_DEVICE_PARTITION_TYPE = 0x1046

const CL_DEVICE_REFERENCE_COUNT = 0x1047

const CL_DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048

const CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049

const CL_DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104a

const CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104b

const CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS = 0x104c

const CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE = 0x104d

const CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES = 0x104e

const CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE = 0x104f

const CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = 0x1050

const CL_DEVICE_MAX_ON_DEVICE_QUEUES = 0x1051

const CL_DEVICE_MAX_ON_DEVICE_EVENTS = 0x1052

const CL_DEVICE_SVM_CAPABILITIES = 0x1053

const CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = 0x1054

const CL_DEVICE_MAX_PIPE_ARGS = 0x1055

const CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056

const CL_DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057

const CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058

const CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059

const CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105a

const CL_DEVICE_IL_VERSION = 0x105b

const CL_DEVICE_MAX_NUM_SUB_GROUPS = 0x105c

const CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105d

const CL_DEVICE_NUMERIC_VERSION = 0x105e

const CL_DEVICE_EXTENSIONS_WITH_VERSION = 0x1060

const CL_DEVICE_ILS_WITH_VERSION = 0x1061

const CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION = 0x1062

const CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES = 0x1063

const CL_DEVICE_ATOMIC_FENCE_CAPABILITIES = 0x1064

const CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT = 0x1065

const CL_DEVICE_OPENCL_C_ALL_VERSIONS = 0x1066

const CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x1067

const CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT = 0x1068

const CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT = 0x1069

const CL_DEVICE_OPENCL_C_FEATURES = 0x106f

const CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES = 0x1070

const CL_DEVICE_PIPE_SUPPORT = 0x1071

const CL_DEVICE_LATEST_CONFORMANCE_VERSION_PASSED = 0x1072

const CL_FP_DENORM = 1 << 0

const CL_FP_INF_NAN = 1 << 1

const CL_FP_ROUND_TO_NEAREST = 1 << 2

const CL_FP_ROUND_TO_ZERO = 1 << 3

const CL_FP_ROUND_TO_INF = 1 << 4

const CL_FP_FMA = 1 << 5

const CL_FP_SOFT_FLOAT = 1 << 6

const CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = 1 << 7

const CL_NONE = 0x00

const CL_READ_ONLY_CACHE = 0x01

const CL_READ_WRITE_CACHE = 0x02

const CL_LOCAL = 0x01

const CL_GLOBAL = 0x02

const CL_EXEC_KERNEL = 1 << 0

const CL_EXEC_NATIVE_KERNEL = 1 << 1

const CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1 << 0

const CL_QUEUE_PROFILING_ENABLE = 1 << 1

const CL_QUEUE_ON_DEVICE = 1 << 2

const CL_QUEUE_ON_DEVICE_DEFAULT = 1 << 3

const CL_CONTEXT_REFERENCE_COUNT = 0x1080

const CL_CONTEXT_DEVICES = 0x1081

const CL_CONTEXT_PROPERTIES = 0x1082

const CL_CONTEXT_NUM_DEVICES = 0x1083

const CL_CONTEXT_PLATFORM = 0x1084

const CL_CONTEXT_INTEROP_USER_SYNC = 0x1085

const CL_DEVICE_PARTITION_EQUALLY = 0x1086

const CL_DEVICE_PARTITION_BY_COUNTS = 0x1087

const CL_DEVICE_PARTITION_BY_COUNTS_LIST_END = 0x00

const CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN = 0x1088

const CL_DEVICE_AFFINITY_DOMAIN_NUMA = 1 << 0

const CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE = 1 << 1

const CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE = 1 << 2

const CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE = 1 << 3

const CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE = 1 << 4

const CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE = 1 << 5

const CL_DEVICE_SVM_COARSE_GRAIN_BUFFER = 1 << 0

const CL_DEVICE_SVM_FINE_GRAIN_BUFFER = 1 << 1

const CL_DEVICE_SVM_FINE_GRAIN_SYSTEM = 1 << 2

const CL_DEVICE_SVM_ATOMICS = 1 << 3

const CL_QUEUE_CONTEXT = 0x1090

const CL_QUEUE_DEVICE = 0x1091

const CL_QUEUE_REFERENCE_COUNT = 0x1092

const CL_QUEUE_PROPERTIES = 0x1093

const CL_QUEUE_SIZE = 0x1094

const CL_QUEUE_DEVICE_DEFAULT = 0x1095

const CL_QUEUE_PROPERTIES_ARRAY = 0x1098

const CL_MEM_READ_WRITE = 1 << 0

const CL_MEM_WRITE_ONLY = 1 << 1

const CL_MEM_READ_ONLY = 1 << 2

const CL_MEM_USE_HOST_PTR = 1 << 3

const CL_MEM_ALLOC_HOST_PTR = 1 << 4

const CL_MEM_COPY_HOST_PTR = 1 << 5

const CL_MEM_HOST_WRITE_ONLY = 1 << 7

const CL_MEM_HOST_READ_ONLY = 1 << 8

const CL_MEM_HOST_NO_ACCESS = 1 << 9

const CL_MEM_SVM_FINE_GRAIN_BUFFER = 1 << 10

const CL_MEM_SVM_ATOMICS = 1 << 11

const CL_MEM_KERNEL_READ_AND_WRITE = 1 << 12

const CL_MIGRATE_MEM_OBJECT_HOST = 1 << 0

const CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED = 1 << 1

const CL_R = 0x10b0

const CL_A = 0x10b1

const CL_RG = 0x10b2

const CL_RA = 0x10b3

const CL_RGB = 0x10b4

const CL_RGBA = 0x10b5

const CL_BGRA = 0x10b6

const CL_ARGB = 0x10b7

const CL_INTENSITY = 0x10b8

const CL_LUMINANCE = 0x10b9

const CL_Rx = 0x10ba

const CL_RGx = 0x10bb

const CL_RGBx = 0x10bc

const CL_DEPTH = 0x10bd

const CL_sRGB = 0x10bf

const CL_sRGBx = 0x10c0

const CL_sRGBA = 0x10c1

const CL_sBGRA = 0x10c2

const CL_ABGR = 0x10c3

const CL_SNORM_INT8 = 0x10d0

const CL_SNORM_INT16 = 0x10d1

const CL_UNORM_INT8 = 0x10d2

const CL_UNORM_INT16 = 0x10d3

const CL_UNORM_SHORT_565 = 0x10d4

const CL_UNORM_SHORT_555 = 0x10d5

const CL_UNORM_INT_101010 = 0x10d6

const CL_SIGNED_INT8 = 0x10d7

const CL_SIGNED_INT16 = 0x10d8

const CL_SIGNED_INT32 = 0x10d9

const CL_UNSIGNED_INT8 = 0x10da

const CL_UNSIGNED_INT16 = 0x10db

const CL_UNSIGNED_INT32 = 0x10dc

const CL_HALF_FLOAT = 0x10dd

const CL_FLOAT = 0x10de

const CL_UNORM_INT_101010_2 = 0x10e0

const CL_MEM_OBJECT_BUFFER = 0x10f0

const CL_MEM_OBJECT_IMAGE2D = 0x10f1

const CL_MEM_OBJECT_IMAGE3D = 0x10f2

const CL_MEM_OBJECT_IMAGE2D_ARRAY = 0x10f3

const CL_MEM_OBJECT_IMAGE1D = 0x10f4

const CL_MEM_OBJECT_IMAGE1D_ARRAY = 0x10f5

const CL_MEM_OBJECT_IMAGE1D_BUFFER = 0x10f6

const CL_MEM_OBJECT_PIPE = 0x10f7

const CL_MEM_TYPE = 0x1100

const CL_MEM_FLAGS = 0x1101

const CL_MEM_SIZE = 0x1102

const CL_MEM_HOST_PTR = 0x1103

const CL_MEM_MAP_COUNT = 0x1104

const CL_MEM_REFERENCE_COUNT = 0x1105

const CL_MEM_CONTEXT = 0x1106

const CL_MEM_ASSOCIATED_MEMOBJECT = 0x1107

const CL_MEM_OFFSET = 0x1108

const CL_MEM_USES_SVM_POINTER = 0x1109

const CL_MEM_PROPERTIES = 0x110a

const CL_IMAGE_FORMAT = 0x1110

const CL_IMAGE_ELEMENT_SIZE = 0x1111

const CL_IMAGE_ROW_PITCH = 0x1112

const CL_IMAGE_SLICE_PITCH = 0x1113

const CL_IMAGE_WIDTH = 0x1114

const CL_IMAGE_HEIGHT = 0x1115

const CL_IMAGE_DEPTH = 0x1116

const CL_IMAGE_ARRAY_SIZE = 0x1117

const CL_IMAGE_BUFFER = 0x1118

const CL_IMAGE_NUM_MIP_LEVELS = 0x1119

const CL_IMAGE_NUM_SAMPLES = 0x111a

const CL_PIPE_PACKET_SIZE = 0x1120

const CL_PIPE_MAX_PACKETS = 0x1121

const CL_PIPE_PROPERTIES = 0x1122

const CL_ADDRESS_NONE = 0x1130

const CL_ADDRESS_CLAMP_TO_EDGE = 0x1131

const CL_ADDRESS_CLAMP = 0x1132

const CL_ADDRESS_REPEAT = 0x1133

const CL_ADDRESS_MIRRORED_REPEAT = 0x1134

const CL_FILTER_NEAREST = 0x1140

const CL_FILTER_LINEAR = 0x1141

const CL_SAMPLER_REFERENCE_COUNT = 0x1150

const CL_SAMPLER_CONTEXT = 0x1151

const CL_SAMPLER_NORMALIZED_COORDS = 0x1152

const CL_SAMPLER_ADDRESSING_MODE = 0x1153

const CL_SAMPLER_FILTER_MODE = 0x1154

const CL_SAMPLER_MIP_FILTER_MODE = 0x1155

const CL_SAMPLER_LOD_MIN = 0x1156

const CL_SAMPLER_LOD_MAX = 0x1157

const CL_SAMPLER_PROPERTIES = 0x1158

const CL_MAP_READ = 1 << 0

const CL_MAP_WRITE = 1 << 1

const CL_MAP_WRITE_INVALIDATE_REGION = 1 << 2

const CL_PROGRAM_REFERENCE_COUNT = 0x1160

const CL_PROGRAM_CONTEXT = 0x1161

const CL_PROGRAM_NUM_DEVICES = 0x1162

const CL_PROGRAM_DEVICES = 0x1163

const CL_PROGRAM_SOURCE = 0x1164

const CL_PROGRAM_BINARY_SIZES = 0x1165

const CL_PROGRAM_BINARIES = 0x1166

const CL_PROGRAM_NUM_KERNELS = 0x1167

const CL_PROGRAM_KERNEL_NAMES = 0x1168

const CL_PROGRAM_IL = 0x1169

const CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT = 0x116a

const CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT = 0x116b

const CL_PROGRAM_BUILD_STATUS = 0x1181

const CL_PROGRAM_BUILD_OPTIONS = 0x1182

const CL_PROGRAM_BUILD_LOG = 0x1183

const CL_PROGRAM_BINARY_TYPE = 0x1184

const CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE = 0x1185

const CL_PROGRAM_BINARY_TYPE_NONE = 0x00

const CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 0x01

const CL_PROGRAM_BINARY_TYPE_LIBRARY = 0x02

const CL_PROGRAM_BINARY_TYPE_EXECUTABLE = 0x04

const CL_BUILD_SUCCESS = 0

const CL_BUILD_NONE = -1

const CL_BUILD_ERROR = -2

const CL_BUILD_IN_PROGRESS = -3

const CL_KERNEL_FUNCTION_NAME = 0x1190

const CL_KERNEL_NUM_ARGS = 0x1191

const CL_KERNEL_REFERENCE_COUNT = 0x1192

const CL_KERNEL_CONTEXT = 0x1193

const CL_KERNEL_PROGRAM = 0x1194

const CL_KERNEL_ATTRIBUTES = 0x1195

const CL_KERNEL_ARG_ADDRESS_QUALIFIER = 0x1196

const CL_KERNEL_ARG_ACCESS_QUALIFIER = 0x1197

const CL_KERNEL_ARG_TYPE_NAME = 0x1198

const CL_KERNEL_ARG_TYPE_QUALIFIER = 0x1199

const CL_KERNEL_ARG_NAME = 0x119a

const CL_KERNEL_ARG_ADDRESS_GLOBAL = 0x119b

const CL_KERNEL_ARG_ADDRESS_LOCAL = 0x119c

const CL_KERNEL_ARG_ADDRESS_CONSTANT = 0x119d

const CL_KERNEL_ARG_ADDRESS_PRIVATE = 0x119e

const CL_KERNEL_ARG_ACCESS_READ_ONLY = 0x11a0

const CL_KERNEL_ARG_ACCESS_WRITE_ONLY = 0x11a1

const CL_KERNEL_ARG_ACCESS_READ_WRITE = 0x11a2

const CL_KERNEL_ARG_ACCESS_NONE = 0x11a3

const CL_KERNEL_ARG_TYPE_NONE = 0

const CL_KERNEL_ARG_TYPE_CONST = 1 << 0

const CL_KERNEL_ARG_TYPE_RESTRICT = 1 << 1

const CL_KERNEL_ARG_TYPE_VOLATILE = 1 << 2

const CL_KERNEL_ARG_TYPE_PIPE = 1 << 3

const CL_KERNEL_WORK_GROUP_SIZE = 0x11b0

const CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11b1

const CL_KERNEL_LOCAL_MEM_SIZE = 0x11b2

const CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11b3

const CL_KERNEL_PRIVATE_MEM_SIZE = 0x11b4

const CL_KERNEL_GLOBAL_WORK_SIZE = 0x11b5

const CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE = 0x2033

const CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE = 0x2034

const CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT = 0x11b8

const CL_KERNEL_MAX_NUM_SUB_GROUPS = 0x11b9

const CL_KERNEL_COMPILE_NUM_SUB_GROUPS = 0x11ba

const CL_KERNEL_EXEC_INFO_SVM_PTRS = 0x11b6

const CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM = 0x11b7

const CL_EVENT_COMMAND_QUEUE = 0x11d0

const CL_EVENT_COMMAND_TYPE = 0x11d1

const CL_EVENT_REFERENCE_COUNT = 0x11d2

const CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11d3

const CL_EVENT_CONTEXT = 0x11d4

const CL_COMMAND_NDRANGE_KERNEL = 0x11f0

const CL_COMMAND_TASK = 0x11f1

const CL_COMMAND_NATIVE_KERNEL = 0x11f2

const CL_COMMAND_READ_BUFFER = 0x11f3

const CL_COMMAND_WRITE_BUFFER = 0x11f4

const CL_COMMAND_COPY_BUFFER = 0x11f5

const CL_COMMAND_READ_IMAGE = 0x11f6

const CL_COMMAND_WRITE_IMAGE = 0x11f7

const CL_COMMAND_COPY_IMAGE = 0x11f8

const CL_COMMAND_COPY_IMAGE_TO_BUFFER = 0x11f9

const CL_COMMAND_COPY_BUFFER_TO_IMAGE = 0x11fa

const CL_COMMAND_MAP_BUFFER = 0x11fb

const CL_COMMAND_MAP_IMAGE = 0x11fc

const CL_COMMAND_UNMAP_MEM_OBJECT = 0x11fd

const CL_COMMAND_MARKER = 0x11fe

const CL_COMMAND_ACQUIRE_GL_OBJECTS = 0x11ff

const CL_COMMAND_RELEASE_GL_OBJECTS = 0x1200

const CL_COMMAND_READ_BUFFER_RECT = 0x1201

const CL_COMMAND_WRITE_BUFFER_RECT = 0x1202

const CL_COMMAND_COPY_BUFFER_RECT = 0x1203

const CL_COMMAND_USER = 0x1204

const CL_COMMAND_BARRIER = 0x1205

const CL_COMMAND_MIGRATE_MEM_OBJECTS = 0x1206

const CL_COMMAND_FILL_BUFFER = 0x1207

const CL_COMMAND_FILL_IMAGE = 0x1208

const CL_COMMAND_SVM_FREE = 0x1209

const CL_COMMAND_SVM_MEMCPY = 0x120a

const CL_COMMAND_SVM_MEMFILL = 0x120b

const CL_COMMAND_SVM_MAP = 0x120c

const CL_COMMAND_SVM_UNMAP = 0x120d

const CL_COMMAND_SVM_MIGRATE_MEM = 0x120e

const CL_COMPLETE = 0x00

const CL_RUNNING = 0x01

const CL_SUBMITTED = 0x02

const CL_QUEUED = 0x03

const CL_BUFFER_CREATE_TYPE_REGION = 0x1220

const CL_PROFILING_COMMAND_QUEUED = 0x1280

const CL_PROFILING_COMMAND_SUBMIT = 0x1281

const CL_PROFILING_COMMAND_START = 0x1282

const CL_PROFILING_COMMAND_END = 0x1283

const CL_PROFILING_COMMAND_COMPLETE = 0x1284

const CL_DEVICE_ATOMIC_ORDER_RELAXED = 1 << 0

const CL_DEVICE_ATOMIC_ORDER_ACQ_REL = 1 << 1

const CL_DEVICE_ATOMIC_ORDER_SEQ_CST = 1 << 2

const CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM = 1 << 3

const CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP = 1 << 4

const CL_DEVICE_ATOMIC_SCOPE_DEVICE = 1 << 5

const CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES = 1 << 6

const CL_DEVICE_QUEUE_SUPPORTED = 1 << 0

const CL_DEVICE_QUEUE_REPLACEABLE_DEFAULT = 1 << 1

const CL_KHRONOS_VENDOR_ID_CODEPLAY = 0x00010004

const CL_VERSION_MAJOR_BITS = 10

const CL_VERSION_MINOR_BITS = 10

const CL_VERSION_PATCH_BITS = 12

const CL_VERSION_MAJOR_MASK = 1 << CL_VERSION_MAJOR_BITS - 1

const CL_VERSION_MINOR_MASK = 1 << CL_VERSION_MINOR_BITS - 1

const CL_VERSION_PATCH_MASK = 1 << CL_VERSION_PATCH_BITS - 1

const cl_khr_gl_sharing = 1

const CL_KHR_GL_SHARING_EXTENSION_NAME = "cl_khr_gl_sharing"

const CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR = -1000

const CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR = 0x2006

const CL_DEVICES_FOR_GL_CONTEXT_KHR = 0x2007

const CL_GL_CONTEXT_KHR = 0x2008

const CL_EGL_DISPLAY_KHR = 0x2009

const CL_GLX_DISPLAY_KHR = 0x200a

const CL_WGL_HDC_KHR = 0x200b

const CL_CGL_SHAREGROUP_KHR = 0x200c

const CL_GL_OBJECT_BUFFER = 0x2000

const CL_GL_OBJECT_TEXTURE2D = 0x2001

const CL_GL_OBJECT_TEXTURE3D = 0x2002

const CL_GL_OBJECT_RENDERBUFFER = 0x2003

const CL_GL_OBJECT_TEXTURE2D_ARRAY = 0x200e

const CL_GL_OBJECT_TEXTURE1D = 0x200f

const CL_GL_OBJECT_TEXTURE1D_ARRAY = 0x2010

const CL_GL_OBJECT_TEXTURE_BUFFER = 0x2011

const CL_GL_TEXTURE_TARGET = 0x2004

const CL_GL_MIPMAP_LEVEL = 0x2005

const cl_khr_gl_event = 1

const CL_KHR_GL_EVENT_EXTENSION_NAME = "cl_khr_gl_event"

const CL_COMMAND_GL_FENCE_SYNC_OBJECT_KHR = 0x200d

const cl_khr_gl_depth_images = 1

const CL_KHR_GL_DEPTH_IMAGES_EXTENSION_NAME = "cl_khr_gl_depth_images"

const CL_DEPTH_STENCIL = 0x10be

const CL_UNORM_INT24 = 0x10df

const cl_khr_gl_msaa_sharing = 1

const CL_KHR_GL_MSAA_SHARING_EXTENSION_NAME = "cl_khr_gl_msaa_sharing"

const CL_GL_NUM_SAMPLES = 0x2012

const cl_intel_sharing_format_query_gl = 1

const CL_INTEL_SHARING_FORMAT_QUERY_GL_EXTENSION_NAME = "cl_intel_sharing_format_query_gl"

const cl_khr_command_buffer = 1

const CL_KHR_COMMAND_BUFFER_EXTENSION_NAME = "cl_khr_command_buffer"

const CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR = 0x12a9

const CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR = 0x12aa

const CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR = 1 << 0

const CL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR = 1 << 1

const CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR = 1 << 2

const CL_COMMAND_BUFFER_CAPABILITY_OUT_OF_ORDER_KHR = 1 << 3

const CL_COMMAND_BUFFER_FLAGS_KHR = 0x1293

const CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR = 1 << 0

const CL_INVALID_COMMAND_BUFFER_KHR = -1138

const CL_INVALID_SYNC_POINT_WAIT_LIST_KHR = -1139

const CL_INCOMPATIBLE_COMMAND_QUEUE_KHR = -1140

const CL_COMMAND_BUFFER_QUEUES_KHR = 0x1294

const CL_COMMAND_BUFFER_NUM_QUEUES_KHR = 0x1295

const CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR = 0x1296

const CL_COMMAND_BUFFER_STATE_KHR = 0x1297

const CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR = 0x1298

const CL_COMMAND_BUFFER_CONTEXT_KHR = 0x1299

const CL_COMMAND_BUFFER_STATE_RECORDING_KHR = 0

const CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR = 1

const CL_COMMAND_BUFFER_STATE_PENDING_KHR = 2

const CL_COMMAND_COMMAND_BUFFER_KHR = 0x12a8

const cl_khr_command_buffer_multi_device = 1

const CL_KHR_COMMAND_BUFFER_MULTI_DEVICE_EXTENSION_NAME = "cl_khr_command_buffer_multi_device"

const CL_PLATFORM_COMMAND_BUFFER_CAPABILITIES_KHR = 0x0908

const CL_COMMAND_BUFFER_PLATFORM_UNIVERSAL_SYNC_KHR = 1 << 0

const CL_COMMAND_BUFFER_PLATFORM_REMAP_QUEUES_KHR = 1 << 1

const CL_COMMAND_BUFFER_PLATFORM_AUTOMATIC_REMAP_KHR = 1 << 2

const CL_DEVICE_COMMAND_BUFFER_NUM_SYNC_DEVICES_KHR = 0x12ab

const CL_DEVICE_COMMAND_BUFFER_SYNC_DEVICES_KHR = 0x12ac

const CL_COMMAND_BUFFER_CAPABILITY_MULTIPLE_QUEUE_KHR = 1 << 4

const CL_COMMAND_BUFFER_DEVICE_SIDE_SYNC_KHR = 1 << 2

const cl_khr_command_buffer_mutable_dispatch = 1

const CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME = "cl_khr_command_buffer_mutable_dispatch"

const CL_COMMAND_BUFFER_MUTABLE_KHR = 1 << 1

const CL_INVALID_MUTABLE_COMMAND_KHR = -1141

const CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR = 0x12b0

const CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR = 0x12b1

const CL_MUTABLE_DISPATCH_GLOBAL_OFFSET_KHR = 1 << 0

const CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR = 1 << 1

const CL_MUTABLE_DISPATCH_LOCAL_SIZE_KHR = 1 << 2

const CL_MUTABLE_DISPATCH_ARGUMENTS_KHR = 1 << 3

const CL_MUTABLE_DISPATCH_EXEC_INFO_KHR = 1 << 4

const CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR = 0x12a0

const CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR = 0x12a1

const CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR = 0x12ad

const CL_MUTABLE_DISPATCH_PROPERTIES_ARRAY_KHR = 0x12a2

const CL_MUTABLE_DISPATCH_KERNEL_KHR = 0x12a3

const CL_MUTABLE_DISPATCH_DIMENSIONS_KHR = 0x12a4

const CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR = 0x12a5

const CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR = 0x12a6

const CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR = 0x12a7

const CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR = 0

const CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR = 1

const CL_COMMAND_BUFFER_MUTABLE_DISPATCH_ASSERTS_KHR = 0x12b7

const CL_MUTABLE_DISPATCH_ASSERTS_KHR = 0x12b8

const CL_MUTABLE_DISPATCH_ASSERT_NO_ADDITIONAL_WORK_GROUPS_KHR = 1 << 0

const cl_khr_fp64 = 1

const CL_KHR_FP64_EXTENSION_NAME = "cl_khr_fp64"

const cl_khr_fp16 = 1

const CL_KHR_FP16_EXTENSION_NAME = "cl_khr_fp16"

const CL_DEVICE_HALF_FP_CONFIG = 0x1033

const cl_APPLE_SetMemObjectDestructor = 1

const CL_APPLE_SETMEMOBJECTDESTRUCTOR_EXTENSION_NAME = "cl_APPLE_SetMemObjectDestructor"

const cl_APPLE_ContextLoggingFunctions = 1

const CL_APPLE_CONTEXTLOGGINGFUNCTIONS_EXTENSION_NAME = "cl_APPLE_ContextLoggingFunctions"

const cl_khr_icd = 1

const CL_KHR_ICD_EXTENSION_NAME = "cl_khr_icd"

const CL_PLATFORM_ICD_SUFFIX_KHR = 0x0920

const CL_PLATFORM_NOT_FOUND_KHR = -1001

const cl_khr_il_program = 1

const CL_KHR_IL_PROGRAM_EXTENSION_NAME = "cl_khr_il_program"

const CL_DEVICE_IL_VERSION_KHR = 0x105b

const CL_PROGRAM_IL_KHR = 0x1169

const cl_khr_image2d_from_buffer = 1

const CL_KHR_IMAGE2D_FROM_BUFFER_EXTENSION_NAME = "cl_khr_image2d_from_buffer"

const CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR = 0x104a

const CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT_KHR = 0x104b

const cl_khr_initialize_memory = 1

const CL_KHR_INITIALIZE_MEMORY_EXTENSION_NAME = "cl_khr_initialize_memory"

const CL_CONTEXT_MEMORY_INITIALIZE_KHR = 0x2030

const CL_CONTEXT_MEMORY_INITIALIZE_LOCAL_KHR = 1 << 0

const CL_CONTEXT_MEMORY_INITIALIZE_PRIVATE_KHR = 1 << 1

const cl_khr_terminate_context = 1

const CL_KHR_TERMINATE_CONTEXT_EXTENSION_NAME = "cl_khr_terminate_context"

const CL_DEVICE_TERMINATE_CAPABILITY_KHR = 0x2031

const CL_CONTEXT_TERMINATE_KHR = 0x2032

const CL_DEVICE_TERMINATE_CAPABILITY_CONTEXT_KHR = 1 << 0

const CL_CONTEXT_TERMINATED_KHR = -1121

const cl_khr_spir = 1

const CL_KHR_SPIR_EXTENSION_NAME = "cl_khr_spir"

const CL_DEVICE_SPIR_VERSIONS = 0x40e0

const CL_PROGRAM_BINARY_TYPE_INTERMEDIATE = 0x40e1

const cl_khr_create_command_queue = 1

const CL_KHR_CREATE_COMMAND_QUEUE_EXTENSION_NAME = "cl_khr_create_command_queue"

const cl_nv_device_attribute_query = 1

const CL_NV_DEVICE_ATTRIBUTE_QUERY_EXTENSION_NAME = "cl_nv_device_attribute_query"

const CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV = 0x4000

const CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV = 0x4001

const CL_DEVICE_REGISTERS_PER_BLOCK_NV = 0x4002

const CL_DEVICE_WARP_SIZE_NV = 0x4003

const CL_DEVICE_GPU_OVERLAP_NV = 0x4004

const CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV = 0x4005

const CL_DEVICE_INTEGRATED_MEMORY_NV = 0x4006

const cl_amd_device_attribute_query = 1

const CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXTENSION_NAME = "cl_amd_device_attribute_query"

const CL_DEVICE_PROFILING_TIMER_OFFSET_AMD = 0x4036

const CL_DEVICE_TOPOLOGY_AMD = 0x4037

const CL_DEVICE_BOARD_NAME_AMD = 0x4038

const CL_DEVICE_GLOBAL_FREE_MEMORY_AMD = 0x4039

const CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD = 0x4040

const CL_DEVICE_SIMD_WIDTH_AMD = 0x4041

const CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD = 0x4042

const CL_DEVICE_WAVEFRONT_WIDTH_AMD = 0x4043

const CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD = 0x4044

const CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD = 0x4045

const CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD = 0x4046

const CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD = 0x4047

const CL_DEVICE_LOCAL_MEM_BANKS_AMD = 0x4048

const CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD = 0x4049

const CL_DEVICE_GFXIP_MAJOR_AMD = 0x404a

const CL_DEVICE_GFXIP_MINOR_AMD = 0x404b

const CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD = 0x404c

const CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD = 0x4030

const CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD = 0x4031

const CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD = 0x4033

const CL_DEVICE_PCIE_ID_AMD = 0x4034

const cl_arm_printf = 1

const CL_ARM_PRINTF_EXTENSION_NAME = "cl_arm_printf"

const CL_PRINTF_CALLBACK_ARM = 0x40b0

const CL_PRINTF_BUFFERSIZE_ARM = 0x40b1

const cl_ext_device_fission = 1

const CL_EXT_DEVICE_FISSION_EXTENSION_NAME = "cl_ext_device_fission"

const CL_DEVICE_PARTITION_FAILED_EXT = -1057

const CL_INVALID_PARTITION_COUNT_EXT = -1058

const CL_INVALID_PARTITION_NAME_EXT = -1059

const CL_DEVICE_PARENT_DEVICE_EXT = 0x4054

const CL_DEVICE_PARTITION_TYPES_EXT = 0x4055

const CL_DEVICE_AFFINITY_DOMAINS_EXT = 0x4056

const CL_DEVICE_REFERENCE_COUNT_EXT = 0x4057

const CL_DEVICE_PARTITION_STYLE_EXT = 0x4058

const CL_DEVICE_PARTITION_EQUALLY_EXT = 0x4050

const CL_DEVICE_PARTITION_BY_COUNTS_EXT = 0x4051

const CL_DEVICE_PARTITION_BY_NAMES_EXT = 0x4052

const CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT = 0x4053

const CL_AFFINITY_DOMAIN_L1_CACHE_EXT = 0x01

const CL_AFFINITY_DOMAIN_L2_CACHE_EXT = 0x02

const CL_AFFINITY_DOMAIN_L3_CACHE_EXT = 0x03

const CL_AFFINITY_DOMAIN_L4_CACHE_EXT = 0x04

const CL_AFFINITY_DOMAIN_NUMA_EXT = 0x10

const CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT = 0x0100

const CL_PROPERTIES_LIST_END_EXT = cl_device_partition_property_ext(0)

const CL_PARTITION_BY_COUNTS_LIST_END_EXT = cl_device_partition_property_ext(0)

const cl_ext_migrate_memobject = 1

const CL_EXT_MIGRATE_MEMOBJECT_EXTENSION_NAME = "cl_ext_migrate_memobject"

const CL_MIGRATE_MEM_OBJECT_HOST_EXT = 1 << 0

const CL_COMMAND_MIGRATE_MEM_OBJECT_EXT = 0x4040

const cl_ext_cxx_for_opencl = 1

const CL_EXT_CXX_FOR_OPENCL_EXTENSION_NAME = "cl_ext_cxx_for_opencl"

const CL_DEVICE_CXX_FOR_OPENCL_NUMERIC_VERSION_EXT = 0x4230

const cl_qcom_ext_host_ptr = 1

const CL_QCOM_EXT_HOST_PTR_EXTENSION_NAME = "cl_qcom_ext_host_ptr"

const CL_MEM_EXT_HOST_PTR_QCOM = 1 << 29

const CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM = 0x40a0

const CL_DEVICE_PAGE_SIZE_QCOM = 0x40a1

const CL_IMAGE_ROW_ALIGNMENT_QCOM = 0x40a2

const CL_IMAGE_SLICE_ALIGNMENT_QCOM = 0x40a3

const CL_MEM_HOST_UNCACHED_QCOM = 0x40a4

const CL_MEM_HOST_WRITEBACK_QCOM = 0x40a5

const CL_MEM_HOST_WRITETHROUGH_QCOM = 0x40a6

const CL_MEM_HOST_WRITE_COMBINING_QCOM = 0x40a7

const cl_qcom_ext_host_ptr_iocoherent = 1

const CL_QCOM_EXT_HOST_PTR_IOCOHERENT_EXTENSION_NAME = "cl_qcom_ext_host_ptr_iocoherent"

const CL_MEM_HOST_IOCOHERENT_QCOM = 0x40a9

const cl_qcom_ion_host_ptr = 1

const CL_QCOM_ION_HOST_PTR_EXTENSION_NAME = "cl_qcom_ion_host_ptr"

const CL_MEM_ION_HOST_PTR_QCOM = 0x40a8

const cl_qcom_android_native_buffer_host_ptr = 1

const CL_QCOM_ANDROID_NATIVE_BUFFER_HOST_PTR_EXTENSION_NAME = "cl_qcom_android_native_buffer_host_ptr"

const CL_MEM_ANDROID_NATIVE_BUFFER_HOST_PTR_QCOM = 0x40c6

const cl_img_yuv_image = 1

const CL_IMG_YUV_IMAGE_EXTENSION_NAME = "cl_img_yuv_image"

const CL_NV21_IMG = 0x40d0

const CL_YV12_IMG = 0x40d1

const cl_img_cached_allocations = 1

const CL_IMG_CACHED_ALLOCATIONS_EXTENSION_NAME = "cl_img_cached_allocations"

const CL_MEM_USE_UNCACHED_CPU_MEMORY_IMG = 1 << 26

const CL_MEM_USE_CACHED_CPU_MEMORY_IMG = 1 << 27

const cl_img_use_gralloc_ptr = 1

const CL_IMG_USE_GRALLOC_PTR_EXTENSION_NAME = "cl_img_use_gralloc_ptr"

const CL_GRALLOC_RESOURCE_NOT_ACQUIRED_IMG = 0x40d4

const CL_INVALID_GRALLOC_OBJECT_IMG = 0x40d5

const CL_MEM_USE_GRALLOC_PTR_IMG = 1 << 28

const CL_COMMAND_ACQUIRE_GRALLOC_OBJECTS_IMG = 0x40d2

const CL_COMMAND_RELEASE_GRALLOC_OBJECTS_IMG = 0x40d3

const cl_img_generate_mipmap = 1

const CL_IMG_GENERATE_MIPMAP_EXTENSION_NAME = "cl_img_generate_mipmap"

const CL_MIPMAP_FILTER_ANY_IMG = 0x00

const CL_MIPMAP_FILTER_BOX_IMG = 0x01

const CL_COMMAND_GENERATE_MIPMAP_IMG = 0x40d6

const cl_img_mem_properties = 1

const CL_IMG_MEM_PROPERTIES_EXTENSION_NAME = "cl_img_mem_properties"

const CL_MEM_ALLOC_FLAGS_IMG = 0x40d7

const CL_MEM_ALLOC_RELAX_REQUIREMENTS_IMG = 1 << 0

const CL_MEM_ALLOC_GPU_WRITE_COMBINE_IMG = 1 << 1

const CL_MEM_ALLOC_GPU_CACHED_IMG = 1 << 2

const CL_MEM_ALLOC_CPU_LOCAL_IMG = 1 << 3

const CL_MEM_ALLOC_GPU_LOCAL_IMG = 1 << 4

const CL_MEM_ALLOC_GPU_PRIVATE_IMG = 1 << 5

const CL_DEVICE_MEMORY_CAPABILITIES_IMG = 0x40d8

const cl_khr_subgroups = 1

const CL_KHR_SUBGROUPS_EXTENSION_NAME = "cl_khr_subgroups"

const CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE_KHR = 0x2033

const CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE_KHR = 0x2034

const cl_khr_mipmap_image = 1

const CL_KHR_MIPMAP_IMAGE_EXTENSION_NAME = "cl_khr_mipmap_image"

const CL_SAMPLER_MIP_FILTER_MODE_KHR = 0x1155

const CL_SAMPLER_LOD_MIN_KHR = 0x1156

const CL_SAMPLER_LOD_MAX_KHR = 0x1157

const cl_khr_priority_hints = 1

const CL_KHR_PRIORITY_HINTS_EXTENSION_NAME = "cl_khr_priority_hints"

const CL_QUEUE_PRIORITY_KHR = 0x1096

const CL_QUEUE_PRIORITY_HIGH_KHR = 1 << 0

const CL_QUEUE_PRIORITY_MED_KHR = 1 << 1

const CL_QUEUE_PRIORITY_LOW_KHR = 1 << 2

const cl_khr_throttle_hints = 1

const CL_KHR_THROTTLE_HINTS_EXTENSION_NAME = "cl_khr_throttle_hints"

const CL_QUEUE_THROTTLE_KHR = 0x1097

const CL_QUEUE_THROTTLE_HIGH_KHR = 1 << 0

const CL_QUEUE_THROTTLE_MED_KHR = 1 << 1

const CL_QUEUE_THROTTLE_LOW_KHR = 1 << 2

const cl_khr_subgroup_named_barrier = 1

const CL_KHR_SUBGROUP_NAMED_BARRIER_EXTENSION_NAME = "cl_khr_subgroup_named_barrier"

const CL_DEVICE_MAX_NAMED_BARRIER_COUNT_KHR = 0x2035

const cl_khr_extended_versioning = 1

const CL_KHR_EXTENDED_VERSIONING_EXTENSION_NAME = "cl_khr_extended_versioning"

const CL_VERSION_MAJOR_BITS_KHR = 10

const CL_VERSION_MINOR_BITS_KHR = 10

const CL_VERSION_PATCH_BITS_KHR = 12

const CL_VERSION_MAJOR_MASK_KHR = 1 << CL_VERSION_MAJOR_BITS_KHR - 1

const CL_VERSION_MINOR_MASK_KHR = 1 << CL_VERSION_MINOR_BITS_KHR - 1

const CL_VERSION_PATCH_MASK_KHR = 1 << CL_VERSION_PATCH_BITS_KHR - 1

const CL_NAME_VERSION_MAX_NAME_SIZE_KHR = 64

const CL_PLATFORM_NUMERIC_VERSION_KHR = 0x0906

const CL_PLATFORM_EXTENSIONS_WITH_VERSION_KHR = 0x0907

const CL_DEVICE_NUMERIC_VERSION_KHR = 0x105e

const CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR = 0x105f

const CL_DEVICE_EXTENSIONS_WITH_VERSION_KHR = 0x1060

const CL_DEVICE_ILS_WITH_VERSION_KHR = 0x1061

const CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION_KHR = 0x1062

const cl_khr_device_uuid = 1

const CL_KHR_DEVICE_UUID_EXTENSION_NAME = "cl_khr_device_uuid"

const CL_UUID_SIZE_KHR = 16

const CL_LUID_SIZE_KHR = 8

const CL_DEVICE_UUID_KHR = 0x106a

const CL_DRIVER_UUID_KHR = 0x106b

const CL_DEVICE_LUID_VALID_KHR = 0x106c

const CL_DEVICE_LUID_KHR = 0x106d

const CL_DEVICE_NODE_MASK_KHR = 0x106e

const cl_khr_pci_bus_info = 1

const CL_KHR_PCI_BUS_INFO_EXTENSION_NAME = "cl_khr_pci_bus_info"

const CL_DEVICE_PCI_BUS_INFO_KHR = 0x410f

const cl_khr_suggested_local_work_size = 1

const CL_KHR_SUGGESTED_LOCAL_WORK_SIZE_EXTENSION_NAME = "cl_khr_suggested_local_work_size"

const cl_khr_integer_dot_product = 1

const CL_KHR_INTEGER_DOT_PRODUCT_EXTENSION_NAME = "cl_khr_integer_dot_product"

const CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR = 1 << 0

const CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR = 1 << 1

const CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR = 0x1073

const CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_8BIT_KHR = 0x1074

const CL_DEVICE_INTEGER_DOT_PRODUCT_ACCELERATION_PROPERTIES_4x8BIT_PACKED_KHR = 0x1075

const cl_khr_external_memory = 1

const CL_KHR_EXTERNAL_MEMORY_EXTENSION_NAME = "cl_khr_external_memory"

const CL_PLATFORM_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR = 0x2044

const CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR = 0x204f

const CL_DEVICE_EXTERNAL_MEMORY_IMPORT_ASSUME_LINEAR_IMAGES_HANDLE_TYPES_KHR = 0x2052

const CL_MEM_DEVICE_HANDLE_LIST_KHR = 0x2051

const CL_MEM_DEVICE_HANDLE_LIST_END_KHR = 0

const CL_COMMAND_ACQUIRE_EXTERNAL_MEM_OBJECTS_KHR = 0x2047

const CL_COMMAND_RELEASE_EXTERNAL_MEM_OBJECTS_KHR = 0x2048

const cl_khr_external_memory_dma_buf = 1

const CL_KHR_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME = "cl_khr_external_memory_dma_buf"

const CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR = 0x2067

const cl_khr_external_memory_dx = 1

const CL_KHR_EXTERNAL_MEMORY_DX_EXTENSION_NAME = "cl_khr_external_memory_dx"

const CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KHR = 0x2063

const CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KMT_KHR = 0x2064

const CL_EXTERNAL_MEMORY_HANDLE_D3D12_HEAP_KHR = 0x2065

const CL_EXTERNAL_MEMORY_HANDLE_D3D12_RESOURCE_KHR = 0x2066

const cl_khr_external_memory_opaque_fd = 1

const CL_KHR_EXTERNAL_MEMORY_OPAQUE_FD_EXTENSION_NAME = "cl_khr_external_memory_opaque_fd"

const CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR = 0x2060

const cl_khr_external_memory_win32 = 1

const CL_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME = "cl_khr_external_memory_win32"

const CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR = 0x2061

const CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR = 0x2062

const cl_khr_external_semaphore = 1

const CL_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME = "cl_khr_external_semaphore"

const CL_PLATFORM_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR = 0x2037

const CL_PLATFORM_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR = 0x2038

const CL_DEVICE_SEMAPHORE_IMPORT_HANDLE_TYPES_KHR = 0x204d

const CL_DEVICE_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR = 0x204e

const CL_SEMAPHORE_EXPORT_HANDLE_TYPES_KHR = 0x203f

const CL_SEMAPHORE_EXPORT_HANDLE_TYPES_LIST_END_KHR = 0

const CL_SEMAPHORE_EXPORTABLE_KHR = 0x2054

const cl_khr_external_semaphore_dx_fence = 1

const CL_KHR_EXTERNAL_SEMAPHORE_DX_FENCE_EXTENSION_NAME = "cl_khr_external_semaphore_dx_fence"

const CL_SEMAPHORE_HANDLE_D3D12_FENCE_KHR = 0x2059

const cl_khr_external_semaphore_opaque_fd = 1

const CL_KHR_EXTERNAL_SEMAPHORE_OPAQUE_FD_EXTENSION_NAME = "cl_khr_external_semaphore_opaque_fd"

const CL_SEMAPHORE_HANDLE_OPAQUE_FD_KHR = 0x2055

const cl_khr_external_semaphore_sync_fd = 1

const CL_KHR_EXTERNAL_SEMAPHORE_SYNC_FD_EXTENSION_NAME = "cl_khr_external_semaphore_sync_fd"

const CL_SEMAPHORE_HANDLE_SYNC_FD_KHR = 0x2058

const cl_khr_external_semaphore_win32 = 1

const CL_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME = "cl_khr_external_semaphore_win32"

const CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KHR = 0x2056

const CL_SEMAPHORE_HANDLE_OPAQUE_WIN32_KMT_KHR = 0x2057

const cl_khr_semaphore = 1

const CL_KHR_SEMAPHORE_EXTENSION_NAME = "cl_khr_semaphore"

const CL_SEMAPHORE_TYPE_BINARY_KHR = 1

const CL_PLATFORM_SEMAPHORE_TYPES_KHR = 0x2036

const CL_DEVICE_SEMAPHORE_TYPES_KHR = 0x204c

const CL_SEMAPHORE_CONTEXT_KHR = 0x2039

const CL_SEMAPHORE_REFERENCE_COUNT_KHR = 0x203a

const CL_SEMAPHORE_PROPERTIES_KHR = 0x203b

const CL_SEMAPHORE_PAYLOAD_KHR = 0x203c

const CL_SEMAPHORE_TYPE_KHR = 0x203d

const CL_SEMAPHORE_DEVICE_HANDLE_LIST_KHR = 0x2053

const CL_SEMAPHORE_DEVICE_HANDLE_LIST_END_KHR = 0

const CL_COMMAND_SEMAPHORE_WAIT_KHR = 0x2042

const CL_COMMAND_SEMAPHORE_SIGNAL_KHR = 0x2043

const CL_INVALID_SEMAPHORE_KHR = -1142

const cl_arm_import_memory = 1

const CL_ARM_IMPORT_MEMORY_EXTENSION_NAME = "cl_arm_import_memory"

const CL_IMPORT_TYPE_ARM = 0x40b2

const CL_IMPORT_TYPE_HOST_ARM = 0x40b3

const CL_IMPORT_TYPE_DMA_BUF_ARM = 0x40b4

const CL_IMPORT_TYPE_PROTECTED_ARM = 0x40b5

const CL_IMPORT_TYPE_ANDROID_HARDWARE_BUFFER_ARM = 0x41e2

const CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM = 0x41e3

const CL_IMPORT_ANDROID_HARDWARE_BUFFER_PLANE_INDEX_ARM = 0x41ef

const CL_IMPORT_ANDROID_HARDWARE_BUFFER_LAYER_INDEX_ARM = 0x41f0

const cl_arm_shared_virtual_memory = 1

const CL_ARM_SHARED_VIRTUAL_MEMORY_EXTENSION_NAME = "cl_arm_shared_virtual_memory"

const CL_DEVICE_SVM_CAPABILITIES_ARM = 0x40b6

const CL_MEM_USES_SVM_POINTER_ARM = 0x40b7

const CL_KERNEL_EXEC_INFO_SVM_PTRS_ARM = 0x40b8

const CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM_ARM = 0x40b9

const CL_COMMAND_SVM_FREE_ARM = 0x40ba

const CL_COMMAND_SVM_MEMCPY_ARM = 0x40bb

const CL_COMMAND_SVM_MEMFILL_ARM = 0x40bc

const CL_COMMAND_SVM_MAP_ARM = 0x40bd

const CL_COMMAND_SVM_UNMAP_ARM = 0x40be

const CL_DEVICE_SVM_COARSE_GRAIN_BUFFER_ARM = 1 << 0

const CL_DEVICE_SVM_FINE_GRAIN_BUFFER_ARM = 1 << 1

const CL_DEVICE_SVM_FINE_GRAIN_SYSTEM_ARM = 1 << 2

const CL_DEVICE_SVM_ATOMICS_ARM = 1 << 3

const CL_MEM_SVM_FINE_GRAIN_BUFFER_ARM = 1 << 10

const CL_MEM_SVM_ATOMICS_ARM = 1 << 11

const cl_arm_get_core_id = 1

const CL_ARM_GET_CORE_ID_EXTENSION_NAME = "cl_arm_get_core_id"

const CL_DEVICE_COMPUTE_UNITS_BITFIELD_ARM = 0x40bf

const cl_arm_job_slot_selection = 1

const CL_ARM_JOB_SLOT_SELECTION_EXTENSION_NAME = "cl_arm_job_slot_selection"

const CL_DEVICE_JOB_SLOTS_ARM = 0x41e0

const CL_QUEUE_JOB_SLOT_ARM = 0x41e1

const cl_arm_scheduling_controls = 1

const CL_ARM_SCHEDULING_CONTROLS_EXTENSION_NAME = "cl_arm_scheduling_controls"

const CL_DEVICE_SCHEDULING_KERNEL_BATCHING_ARM = 1 << 0

const CL_DEVICE_SCHEDULING_WORKGROUP_BATCH_SIZE_ARM = 1 << 1

const CL_DEVICE_SCHEDULING_WORKGROUP_BATCH_SIZE_MODIFIER_ARM = 1 << 2

const CL_DEVICE_SCHEDULING_DEFERRED_FLUSH_ARM = 1 << 3

const CL_DEVICE_SCHEDULING_REGISTER_ALLOCATION_ARM = 1 << 4

const CL_DEVICE_SCHEDULING_WARP_THROTTLING_ARM = 1 << 5

const CL_DEVICE_SCHEDULING_COMPUTE_UNIT_BATCH_QUEUE_SIZE_ARM = 1 << 6

const CL_DEVICE_SCHEDULING_COMPUTE_UNIT_LIMIT_ARM = 1 << 7

const CL_DEVICE_SCHEDULING_CONTROLS_CAPABILITIES_ARM = 0x41e4

const CL_DEVICE_SUPPORTED_REGISTER_ALLOCATIONS_ARM = 0x41eb

const CL_DEVICE_MAX_WARP_COUNT_ARM = 0x41ea

const CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_ARM = 0x41e5

const CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_MODIFIER_ARM = 0x41e6

const CL_KERNEL_EXEC_INFO_WARP_COUNT_LIMIT_ARM = 0x41e8

const CL_KERNEL_EXEC_INFO_COMPUTE_UNIT_MAX_QUEUED_BATCHES_ARM = 0x41f1

const CL_KERNEL_MAX_WARP_COUNT_ARM = 0x41e9

const CL_QUEUE_KERNEL_BATCHING_ARM = 0x41e7

const CL_QUEUE_DEFERRED_FLUSH_ARM = 0x41ec

const CL_QUEUE_COMPUTE_UNIT_LIMIT_ARM = 0x41f3

const cl_arm_controlled_kernel_termination = 1

const CL_ARM_CONTROLLED_KERNEL_TERMINATION_EXTENSION_NAME = "cl_arm_controlled_kernel_termination"

const CL_COMMAND_TERMINATED_ITSELF_WITH_FAILURE_ARM = -1108

const CL_DEVICE_CONTROLLED_TERMINATION_SUCCESS_ARM = 1 << 0

const CL_DEVICE_CONTROLLED_TERMINATION_FAILURE_ARM = 1 << 1

const CL_DEVICE_CONTROLLED_TERMINATION_QUERY_ARM = 1 << 2

const CL_DEVICE_CONTROLLED_TERMINATION_CAPABILITIES_ARM = 0x41ee

const CL_EVENT_COMMAND_TERMINATION_REASON_ARM = 0x41ed

const CL_COMMAND_TERMINATION_COMPLETION_ARM = 0

const CL_COMMAND_TERMINATION_CONTROLLED_SUCCESS_ARM = 1

const CL_COMMAND_TERMINATION_CONTROLLED_FAILURE_ARM = 2

const CL_COMMAND_TERMINATION_ERROR_ARM = 3

const cl_arm_protected_memory_allocation = 1

const CL_ARM_PROTECTED_MEMORY_ALLOCATION_EXTENSION_NAME = "cl_arm_protected_memory_allocation"

const CL_MEM_PROTECTED_ALLOC_ARM = cl_bitfield(1) << 36

const cl_intel_exec_by_local_thread = 1

const CL_INTEL_EXEC_BY_LOCAL_THREAD_EXTENSION_NAME = "cl_intel_exec_by_local_thread"

const CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL = cl_bitfield(1) << 31

const cl_intel_device_attribute_query = 1

const CL_INTEL_DEVICE_ATTRIBUTE_QUERY_EXTENSION_NAME = "cl_intel_device_attribute_query"

const CL_DEVICE_FEATURE_FLAG_DP4A_INTEL = 1 << 0

const CL_DEVICE_FEATURE_FLAG_DPAS_INTEL = 1 << 1

const CL_DEVICE_IP_VERSION_INTEL = 0x4250

const CL_DEVICE_ID_INTEL = 0x4251

const CL_DEVICE_NUM_SLICES_INTEL = 0x4252

const CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL = 0x4253

const CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL = 0x4254

const CL_DEVICE_NUM_THREADS_PER_EU_INTEL = 0x4255

const CL_DEVICE_FEATURE_CAPABILITIES_INTEL = 0x4256

const cl_intel_device_partition_by_names = 1

const CL_INTEL_DEVICE_PARTITION_BY_NAMES_EXTENSION_NAME = "cl_intel_device_partition_by_names"

const CL_DEVICE_PARTITION_BY_NAMES_INTEL = 0x4052

const CL_PARTITION_BY_NAMES_LIST_END_INTEL = -1

const cl_intel_accelerator = 1

const CL_INTEL_ACCELERATOR_EXTENSION_NAME = "cl_intel_accelerator"

const CL_ACCELERATOR_DESCRIPTOR_INTEL = 0x4090

const CL_ACCELERATOR_REFERENCE_COUNT_INTEL = 0x4091

const CL_ACCELERATOR_CONTEXT_INTEL = 0x4092

const CL_ACCELERATOR_TYPE_INTEL = 0x4093

const CL_INVALID_ACCELERATOR_INTEL = -1094

const CL_INVALID_ACCELERATOR_TYPE_INTEL = -1095

const CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL = -1096

const CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL = -1097

const cl_intel_motion_estimation = 1

const CL_INTEL_MOTION_ESTIMATION_EXTENSION_NAME = "cl_intel_motion_estimation"

const CL_ACCELERATOR_TYPE_MOTION_ESTIMATION_INTEL = 0x00

const CL_ME_MB_TYPE_16x16_INTEL = 0x00

const CL_ME_MB_TYPE_8x8_INTEL = 0x01

const CL_ME_MB_TYPE_4x4_INTEL = 0x02

const CL_ME_SUBPIXEL_MODE_INTEGER_INTEL = 0x00

const CL_ME_SUBPIXEL_MODE_HPEL_INTEL = 0x01

const CL_ME_SUBPIXEL_MODE_QPEL_INTEL = 0x02

const CL_ME_SAD_ADJUST_MODE_NONE_INTEL = 0x00

const CL_ME_SAD_ADJUST_MODE_HAAR_INTEL = 0x01

const CL_ME_SEARCH_PATH_RADIUS_2_2_INTEL = 0x00

const CL_ME_SEARCH_PATH_RADIUS_4_4_INTEL = 0x01

const CL_ME_SEARCH_PATH_RADIUS_16_12_INTEL = 0x05

const cl_intel_advanced_motion_estimation = 1

const CL_INTEL_ADVANCED_MOTION_ESTIMATION_EXTENSION_NAME = "cl_intel_advanced_motion_estimation"

const CL_DEVICE_ME_VERSION_INTEL = 0x407e

const CL_ME_VERSION_LEGACY_INTEL = 0x00

const CL_ME_VERSION_ADVANCED_VER_1_INTEL = 0x01

const CL_ME_VERSION_ADVANCED_VER_2_INTEL = 0x02

const CL_ME_CHROMA_INTRA_PREDICT_ENABLED_INTEL = 0x01

const CL_ME_LUMA_INTRA_PREDICT_ENABLED_INTEL = 0x02

const CL_ME_SKIP_BLOCK_TYPE_16x16_INTEL = 0x00

const CL_ME_SKIP_BLOCK_TYPE_8x8_INTEL = 0x04

const CL_ME_COST_PENALTY_NONE_INTEL = 0x00

const CL_ME_COST_PENALTY_LOW_INTEL = 0x01

const CL_ME_COST_PENALTY_NORMAL_INTEL = 0x02

const CL_ME_COST_PENALTY_HIGH_INTEL = 0x03

const CL_ME_COST_PRECISION_QPEL_INTEL = 0x00

const CL_ME_COST_PRECISION_HPEL_INTEL = 0x01

const CL_ME_COST_PRECISION_PEL_INTEL = 0x02

const CL_ME_COST_PRECISION_DPEL_INTEL = 0x03

const CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_INTEL = 0x00

const CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_INTEL = 0x01

const CL_ME_LUMA_PREDICTOR_MODE_DC_INTEL = 0x02

const CL_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_LEFT_INTEL = 0x03

const CL_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_RIGHT_INTEL = 0x04

const CL_ME_LUMA_PREDICTOR_MODE_PLANE_INTEL = 0x04

const CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_RIGHT_INTEL = 0x05

const CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_DOWN_INTEL = 0x06

const CL_ME_LUMA_PREDICTOR_MODE_VERTICAL_LEFT_INTEL = 0x07

const CL_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_UP_INTEL = 0x08

const CL_ME_CHROMA_PREDICTOR_MODE_DC_INTEL = 0x00

const CL_ME_CHROMA_PREDICTOR_MODE_HORIZONTAL_INTEL = 0x01

const CL_ME_CHROMA_PREDICTOR_MODE_VERTICAL_INTEL = 0x02

const CL_ME_CHROMA_PREDICTOR_MODE_PLANE_INTEL = 0x03

const CL_ME_FORWARD_INPUT_MODE_INTEL = 0x01

const CL_ME_BACKWARD_INPUT_MODE_INTEL = 0x02

const CL_ME_BIDIRECTION_INPUT_MODE_INTEL = 0x03

const CL_ME_BIDIR_WEIGHT_QUARTER_INTEL = 16

const CL_ME_BIDIR_WEIGHT_THIRD_INTEL = 21

const CL_ME_BIDIR_WEIGHT_HALF_INTEL = 32

const CL_ME_BIDIR_WEIGHT_TWO_THIRD_INTEL = 43

const CL_ME_BIDIR_WEIGHT_THREE_QUARTER_INTEL = 48

const cl_intel_simultaneous_sharing = 1

const CL_INTEL_SIMULTANEOUS_SHARING_EXTENSION_NAME = "cl_intel_simultaneous_sharing"

const CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL = 0x4104

const CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL = 0x4105

const cl_intel_egl_image_yuv = 1

const CL_INTEL_EGL_IMAGE_YUV_EXTENSION_NAME = "cl_intel_egl_image_yuv"

const CL_EGL_YUV_PLANE_INTEL = 0x4107

const cl_intel_packed_yuv = 1

const CL_INTEL_PACKED_YUV_EXTENSION_NAME = "cl_intel_packed_yuv"

const CL_YUYV_INTEL = 0x4076

const CL_UYVY_INTEL = 0x4077

const CL_YVYU_INTEL = 0x4078

const CL_VYUY_INTEL = 0x4079

const cl_intel_required_subgroup_size = 1

const CL_INTEL_REQUIRED_SUBGROUP_SIZE_EXTENSION_NAME = "cl_intel_required_subgroup_size"

const CL_DEVICE_SUB_GROUP_SIZES_INTEL = 0x4108

const CL_KERNEL_SPILL_MEM_SIZE_INTEL = 0x4109

const CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL = 0x410a

const cl_intel_driver_diagnostics = 1

const CL_INTEL_DRIVER_DIAGNOSTICS_EXTENSION_NAME = "cl_intel_driver_diagnostics"

const CL_CONTEXT_SHOW_DIAGNOSTICS_INTEL = 0x4106

const CL_CONTEXT_DIAGNOSTICS_LEVEL_ALL_INTEL = 0xff

const CL_CONTEXT_DIAGNOSTICS_LEVEL_GOOD_INTEL = 1 << 0

const CL_CONTEXT_DIAGNOSTICS_LEVEL_BAD_INTEL = 1 << 1

const CL_CONTEXT_DIAGNOSTICS_LEVEL_NEUTRAL_INTEL = 1 << 2

const cl_intel_planar_yuv = 1

const CL_INTEL_PLANAR_YUV_EXTENSION_NAME = "cl_intel_planar_yuv"

const CL_NV12_INTEL = 0x410e

const CL_MEM_NO_ACCESS_INTEL = 1 << 24

const CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL = 1 << 25

const CL_DEVICE_PLANAR_YUV_MAX_WIDTH_INTEL = 0x417e

const CL_DEVICE_PLANAR_YUV_MAX_HEIGHT_INTEL = 0x417f

const cl_intel_device_side_avc_motion_estimation = 1

const CL_INTEL_DEVICE_SIDE_AVC_MOTION_ESTIMATION_EXTENSION_NAME = "cl_intel_device_side_avc_motion_estimation"

const CL_DEVICE_AVC_ME_VERSION_INTEL = 0x410b

const CL_DEVICE_AVC_ME_SUPPORTS_TEXTURE_SAMPLER_USE_INTEL = 0x410c

const CL_DEVICE_AVC_ME_SUPPORTS_PREEMPTION_INTEL = 0x410d

const CL_AVC_ME_VERSION_0_INTEL = 0x00

const CL_AVC_ME_VERSION_1_INTEL = 0x01

const CL_AVC_ME_MAJOR_16x16_INTEL = 0x00

const CL_AVC_ME_MAJOR_16x8_INTEL = 0x01

const CL_AVC_ME_MAJOR_8x16_INTEL = 0x02

const CL_AVC_ME_MAJOR_8x8_INTEL = 0x03

const CL_AVC_ME_MINOR_8x8_INTEL = 0x00

const CL_AVC_ME_MINOR_8x4_INTEL = 0x01

const CL_AVC_ME_MINOR_4x8_INTEL = 0x02

const CL_AVC_ME_MINOR_4x4_INTEL = 0x03

const CL_AVC_ME_MAJOR_FORWARD_INTEL = 0x00

const CL_AVC_ME_MAJOR_BACKWARD_INTEL = 0x01

const CL_AVC_ME_MAJOR_BIDIRECTIONAL_INTEL = 0x02

const CL_AVC_ME_PARTITION_MASK_ALL_INTEL = 0x00

const CL_AVC_ME_PARTITION_MASK_16x16_INTEL = 0x7e

const CL_AVC_ME_PARTITION_MASK_16x8_INTEL = 0x7d

const CL_AVC_ME_PARTITION_MASK_8x16_INTEL = 0x7b

const CL_AVC_ME_PARTITION_MASK_8x8_INTEL = 0x77

const CL_AVC_ME_PARTITION_MASK_8x4_INTEL = 0x6f

const CL_AVC_ME_PARTITION_MASK_4x8_INTEL = 0x5f

const CL_AVC_ME_PARTITION_MASK_4x4_INTEL = 0x3f

const CL_AVC_ME_SEARCH_WINDOW_EXHAUSTIVE_INTEL = 0x00

const CL_AVC_ME_SEARCH_WINDOW_SMALL_INTEL = 0x01

const CL_AVC_ME_SEARCH_WINDOW_TINY_INTEL = 0x02

const CL_AVC_ME_SEARCH_WINDOW_EXTRA_TINY_INTEL = 0x03

const CL_AVC_ME_SEARCH_WINDOW_DIAMOND_INTEL = 0x04

const CL_AVC_ME_SEARCH_WINDOW_LARGE_DIAMOND_INTEL = 0x05

const CL_AVC_ME_SEARCH_WINDOW_RESERVED0_INTEL = 0x06

const CL_AVC_ME_SEARCH_WINDOW_RESERVED1_INTEL = 0x07

const CL_AVC_ME_SEARCH_WINDOW_CUSTOM_INTEL = 0x08

const CL_AVC_ME_SEARCH_WINDOW_16x12_RADIUS_INTEL = 0x09

const CL_AVC_ME_SEARCH_WINDOW_4x4_RADIUS_INTEL = 0x02

const CL_AVC_ME_SEARCH_WINDOW_2x2_RADIUS_INTEL = 0x0a

const CL_AVC_ME_SAD_ADJUST_MODE_NONE_INTEL = 0x00

const CL_AVC_ME_SAD_ADJUST_MODE_HAAR_INTEL = 0x02

const CL_AVC_ME_SUBPIXEL_MODE_INTEGER_INTEL = 0x00

const CL_AVC_ME_SUBPIXEL_MODE_HPEL_INTEL = 0x01

const CL_AVC_ME_SUBPIXEL_MODE_QPEL_INTEL = 0x03

const CL_AVC_ME_COST_PRECISION_QPEL_INTEL = 0x00

const CL_AVC_ME_COST_PRECISION_HPEL_INTEL = 0x01

const CL_AVC_ME_COST_PRECISION_PEL_INTEL = 0x02

const CL_AVC_ME_COST_PRECISION_DPEL_INTEL = 0x03

const CL_AVC_ME_BIDIR_WEIGHT_QUARTER_INTEL = 0x10

const CL_AVC_ME_BIDIR_WEIGHT_THIRD_INTEL = 0x15

const CL_AVC_ME_BIDIR_WEIGHT_HALF_INTEL = 0x20

const CL_AVC_ME_BIDIR_WEIGHT_TWO_THIRD_INTEL = 0x2b

const CL_AVC_ME_BIDIR_WEIGHT_THREE_QUARTER_INTEL = 0x30

const CL_AVC_ME_BORDER_REACHED_LEFT_INTEL = 0x00

const CL_AVC_ME_BORDER_REACHED_RIGHT_INTEL = 0x02

const CL_AVC_ME_BORDER_REACHED_TOP_INTEL = 0x04

const CL_AVC_ME_BORDER_REACHED_BOTTOM_INTEL = 0x08

const CL_AVC_ME_SKIP_BLOCK_PARTITION_16x16_INTEL = 0x00

const CL_AVC_ME_SKIP_BLOCK_PARTITION_8x8_INTEL = 0x4000

const CL_AVC_ME_SKIP_BLOCK_16x16_FORWARD_ENABLE_INTEL = 0x01 << 24

const CL_AVC_ME_SKIP_BLOCK_16x16_BACKWARD_ENABLE_INTEL = 0x02 << 24

const CL_AVC_ME_SKIP_BLOCK_16x16_DUAL_ENABLE_INTEL = 0x03 << 24

const CL_AVC_ME_SKIP_BLOCK_8x8_FORWARD_ENABLE_INTEL = 0x55 << 24

const CL_AVC_ME_SKIP_BLOCK_8x8_BACKWARD_ENABLE_INTEL = 0xaa << 24

const CL_AVC_ME_SKIP_BLOCK_8x8_DUAL_ENABLE_INTEL = 0xff << 24

const CL_AVC_ME_SKIP_BLOCK_8x8_0_FORWARD_ENABLE_INTEL = 0x01 << 24

const CL_AVC_ME_SKIP_BLOCK_8x8_0_BACKWARD_ENABLE_INTEL = 0x02 << 24

const CL_AVC_ME_SKIP_BLOCK_8x8_1_FORWARD_ENABLE_INTEL = 0x01 << 26

const CL_AVC_ME_SKIP_BLOCK_8x8_1_BACKWARD_ENABLE_INTEL = 0x02 << 26

const CL_AVC_ME_SKIP_BLOCK_8x8_2_FORWARD_ENABLE_INTEL = 0x01 << 28

const CL_AVC_ME_SKIP_BLOCK_8x8_2_BACKWARD_ENABLE_INTEL = 0x02 << 28

const CL_AVC_ME_SKIP_BLOCK_8x8_3_FORWARD_ENABLE_INTEL = 0x01 << 30

const CL_AVC_ME_SKIP_BLOCK_8x8_3_BACKWARD_ENABLE_INTEL = 0x02 << 30

const CL_AVC_ME_BLOCK_BASED_SKIP_4x4_INTEL = 0x00

const CL_AVC_ME_BLOCK_BASED_SKIP_8x8_INTEL = 0x80

const CL_AVC_ME_INTRA_16x16_INTEL = 0x00

const CL_AVC_ME_INTRA_8x8_INTEL = 0x01

const CL_AVC_ME_INTRA_4x4_INTEL = 0x02

const CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_16x16_INTEL = 0x06

const CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_8x8_INTEL = 0x05

const CL_AVC_ME_INTRA_LUMA_PARTITION_MASK_4x4_INTEL = 0x03

const CL_AVC_ME_INTRA_NEIGHBOR_LEFT_MASK_ENABLE_INTEL = 0x60

const CL_AVC_ME_INTRA_NEIGHBOR_UPPER_MASK_ENABLE_INTEL = 0x10

const CL_AVC_ME_INTRA_NEIGHBOR_UPPER_RIGHT_MASK_ENABLE_INTEL = 0x08

const CL_AVC_ME_INTRA_NEIGHBOR_UPPER_LEFT_MASK_ENABLE_INTEL = 0x04

const CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_INTEL = 0x00

const CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_INTEL = 0x01

const CL_AVC_ME_LUMA_PREDICTOR_MODE_DC_INTEL = 0x02

const CL_AVC_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_LEFT_INTEL = 0x03

const CL_AVC_ME_LUMA_PREDICTOR_MODE_DIAGONAL_DOWN_RIGHT_INTEL = 0x04

const CL_AVC_ME_LUMA_PREDICTOR_MODE_PLANE_INTEL = 0x04

const CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_RIGHT_INTEL = 0x05

const CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_DOWN_INTEL = 0x06

const CL_AVC_ME_LUMA_PREDICTOR_MODE_VERTICAL_LEFT_INTEL = 0x07

const CL_AVC_ME_LUMA_PREDICTOR_MODE_HORIZONTAL_UP_INTEL = 0x08

const CL_AVC_ME_CHROMA_PREDICTOR_MODE_DC_INTEL = 0x00

const CL_AVC_ME_CHROMA_PREDICTOR_MODE_HORIZONTAL_INTEL = 0x01

const CL_AVC_ME_CHROMA_PREDICTOR_MODE_VERTICAL_INTEL = 0x02

const CL_AVC_ME_CHROMA_PREDICTOR_MODE_PLANE_INTEL = 0x03

const CL_AVC_ME_FRAME_FORWARD_INTEL = 0x01

const CL_AVC_ME_FRAME_BACKWARD_INTEL = 0x02

const CL_AVC_ME_FRAME_DUAL_INTEL = 0x03

const CL_AVC_ME_SLICE_TYPE_PRED_INTEL = 0x00

const CL_AVC_ME_SLICE_TYPE_BPRED_INTEL = 0x01

const CL_AVC_ME_SLICE_TYPE_INTRA_INTEL = 0x02

const CL_AVC_ME_INTERLACED_SCAN_TOP_FIELD_INTEL = 0x00

const CL_AVC_ME_INTERLACED_SCAN_BOTTOM_FIELD_INTEL = 0x01

const cl_intel_unified_shared_memory = 1

const CL_INTEL_UNIFIED_SHARED_MEMORY_EXTENSION_NAME = "cl_intel_unified_shared_memory"

const CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL = 0x4190

const CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL = 0x4191

const CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL = 0x4192

const CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL = 0x4193

const CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL = 0x4194

const CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL = 1 << 0

const CL_UNIFIED_SHARED_MEMORY_ATOMIC_ACCESS_INTEL = 1 << 1

const CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ACCESS_INTEL = 1 << 2

const CL_UNIFIED_SHARED_MEMORY_CONCURRENT_ATOMIC_ACCESS_INTEL = 1 << 3

const CL_MEM_ALLOC_FLAGS_INTEL = 0x4195

const CL_MEM_ALLOC_WRITE_COMBINED_INTEL = 1 << 0

const CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL = 1 << 1

const CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL = 1 << 2

const CL_MEM_ALLOC_TYPE_INTEL = 0x419a

const CL_MEM_ALLOC_BASE_PTR_INTEL = 0x419b

const CL_MEM_ALLOC_SIZE_INTEL = 0x419c

const CL_MEM_ALLOC_DEVICE_INTEL = 0x419d

const CL_MEM_TYPE_UNKNOWN_INTEL = 0x4196

const CL_MEM_TYPE_HOST_INTEL = 0x4197

const CL_MEM_TYPE_DEVICE_INTEL = 0x4198

const CL_MEM_TYPE_SHARED_INTEL = 0x4199

const CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL = 0x4200

const CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL = 0x4201

const CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL = 0x4202

const CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL = 0x4203

const CL_COMMAND_MEMFILL_INTEL = 0x4204

const CL_COMMAND_MEMCPY_INTEL = 0x4205

const CL_COMMAND_MIGRATEMEM_INTEL = 0x4206

const CL_COMMAND_MEMADVISE_INTEL = 0x4207

const cl_intel_mem_alloc_buffer_location = 1

const CL_INTEL_MEM_ALLOC_BUFFER_LOCATION_EXTENSION_NAME = "cl_intel_mem_alloc_buffer_location"

const CL_MEM_ALLOC_BUFFER_LOCATION_INTEL = 0x419e

const cl_intel_create_buffer_with_properties = 1

const CL_INTEL_CREATE_BUFFER_WITH_PROPERTIES_EXTENSION_NAME = "cl_intel_create_buffer_with_properties"

const cl_intel_program_scope_host_pipe = 1

const CL_INTEL_PROGRAM_SCOPE_HOST_PIPE_EXTENSION_NAME = "cl_intel_program_scope_host_pipe"

const CL_COMMAND_READ_HOST_PIPE_INTEL = 0x4214

const CL_COMMAND_WRITE_HOST_PIPE_INTEL = 0x4215

const CL_PROGRAM_NUM_HOST_PIPES_INTEL = 0x4216

const CL_PROGRAM_HOST_PIPE_NAMES_INTEL = 0x4217

const cl_intel_mem_channel_property = 1

const CL_INTEL_MEM_CHANNEL_PROPERTY_EXTENSION_NAME = "cl_intel_mem_channel_property"

const CL_MEM_CHANNEL_INTEL = 0x4213

const cl_intel_mem_force_host_memory = 1

const CL_INTEL_MEM_FORCE_HOST_MEMORY_EXTENSION_NAME = "cl_intel_mem_force_host_memory"

const CL_MEM_FORCE_HOST_MEMORY_INTEL = 1 << 20

const cl_intel_command_queue_families = 1

const CL_INTEL_COMMAND_QUEUE_FAMILIES_EXTENSION_NAME = "cl_intel_command_queue_families"

const CL_QUEUE_FAMILY_MAX_NAME_SIZE_INTEL = 64

const CL_DEVICE_QUEUE_FAMILY_PROPERTIES_INTEL = 0x418b

const CL_QUEUE_FAMILY_INTEL = 0x418c

const CL_QUEUE_INDEX_INTEL = 0x418d

const CL_QUEUE_DEFAULT_CAPABILITIES_INTEL = 0

const CL_QUEUE_CAPABILITY_CREATE_SINGLE_QUEUE_EVENTS_INTEL = 1 << 0

const CL_QUEUE_CAPABILITY_CREATE_CROSS_QUEUE_EVENTS_INTEL = 1 << 1

const CL_QUEUE_CAPABILITY_SINGLE_QUEUE_EVENT_WAIT_LIST_INTEL = 1 << 2

const CL_QUEUE_CAPABILITY_CROSS_QUEUE_EVENT_WAIT_LIST_INTEL = 1 << 3

const CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_INTEL = 1 << 8

const CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_RECT_INTEL = 1 << 9

const CL_QUEUE_CAPABILITY_MAP_BUFFER_INTEL = 1 << 10

const CL_QUEUE_CAPABILITY_FILL_BUFFER_INTEL = 1 << 11

const CL_QUEUE_CAPABILITY_TRANSFER_IMAGE_INTEL = 1 << 12

const CL_QUEUE_CAPABILITY_MAP_IMAGE_INTEL = 1 << 13

const CL_QUEUE_CAPABILITY_FILL_IMAGE_INTEL = 1 << 14

const CL_QUEUE_CAPABILITY_TRANSFER_BUFFER_IMAGE_INTEL = 1 << 15

const CL_QUEUE_CAPABILITY_TRANSFER_IMAGE_BUFFER_INTEL = 1 << 16

const CL_QUEUE_CAPABILITY_MARKER_INTEL = 1 << 24

const CL_QUEUE_CAPABILITY_BARRIER_INTEL = 1 << 25

const CL_QUEUE_CAPABILITY_KERNEL_INTEL = 1 << 26

const cl_intel_queue_no_sync_operations = 1

const CL_INTEL_QUEUE_NO_SYNC_OPERATIONS_EXTENSION_NAME = "cl_intel_queue_no_sync_operations"

const CL_QUEUE_NO_SYNC_OPERATIONS_INTEL = 1 << 29

const cl_intel_sharing_format_query = 1

const CL_INTEL_SHARING_FORMAT_QUERY_EXTENSION_NAME = "cl_intel_sharing_format_query"

const cl_ext_image_requirements_info = 1

const CL_EXT_IMAGE_REQUIREMENTS_INFO_EXTENSION_NAME = "cl_ext_image_requirements_info"

const CL_IMAGE_REQUIREMENTS_BASE_ADDRESS_ALIGNMENT_EXT = 0x1292

const CL_IMAGE_REQUIREMENTS_ROW_PITCH_ALIGNMENT_EXT = 0x1290

const CL_IMAGE_REQUIREMENTS_SIZE_EXT = 0x12b2

const CL_IMAGE_REQUIREMENTS_MAX_WIDTH_EXT = 0x12b3

const CL_IMAGE_REQUIREMENTS_MAX_HEIGHT_EXT = 0x12b4

const CL_IMAGE_REQUIREMENTS_MAX_DEPTH_EXT = 0x12b5

const CL_IMAGE_REQUIREMENTS_MAX_ARRAY_SIZE_EXT = 0x12b6

const cl_ext_image_from_buffer = 1

const CL_EXT_IMAGE_FROM_BUFFER_EXTENSION_NAME = "cl_ext_image_from_buffer"

const CL_IMAGE_REQUIREMENTS_SLICE_PITCH_ALIGNMENT_EXT = 0x1291

const cl_loader_info = 1

const CL_LOADER_INFO_EXTENSION_NAME = "cl_loader_info"

const CL_ICDL_OCL_VERSION = 1

const CL_ICDL_VERSION = 2

const CL_ICDL_NAME = 3

const CL_ICDL_VENDOR = 4

const cl_khr_depth_images = 1

const CL_KHR_DEPTH_IMAGES_EXTENSION_NAME = "cl_khr_depth_images"

const cl_ext_float_atomics = 1

const CL_EXT_FLOAT_ATOMICS_EXTENSION_NAME = "cl_ext_float_atomics"

const CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT = 1 << 0

const CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT = 1 << 1

const CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT = 1 << 2

const CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT = 1 << 16

const CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT = 1 << 17

const CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT = 1 << 18

const CL_DEVICE_SINGLE_FP_ATOMIC_CAPABILITIES_EXT = 0x4231

const CL_DEVICE_DOUBLE_FP_ATOMIC_CAPABILITIES_EXT = 0x4232

const CL_DEVICE_HALF_FP_ATOMIC_CAPABILITIES_EXT = 0x4233

const cl_intel_create_mem_object_properties = 1

const CL_INTEL_CREATE_MEM_OBJECT_PROPERTIES_EXTENSION_NAME = "cl_intel_create_mem_object_properties"

const CL_MEM_LOCALLY_UNCACHED_RESOURCE_INTEL = 0x4218

const CL_MEM_DEVICE_ID_INTEL = 0x4219

const cl_pocl_content_size = 1

const CL_POCL_CONTENT_SIZE_EXTENSION_NAME = "cl_pocl_content_size"

const cl_ext_image_raw10_raw12 = 1

const CL_EXT_IMAGE_RAW10_RAW12_EXTENSION_NAME = "cl_ext_image_raw10_raw12"

const CL_UNSIGNED_INT_RAW10_EXT = 0x10e3

const CL_UNSIGNED_INT_RAW12_EXT = 0x10e4

const cl_khr_3d_image_writes = 1

const CL_KHR_3D_IMAGE_WRITES_EXTENSION_NAME = "cl_khr_3d_image_writes"

const cl_khr_async_work_group_copy_fence = 1

const CL_KHR_ASYNC_WORK_GROUP_COPY_FENCE_EXTENSION_NAME = "cl_khr_async_work_group_copy_fence"

const cl_khr_byte_addressable_store = 1

const CL_KHR_BYTE_ADDRESSABLE_STORE_EXTENSION_NAME = "cl_khr_byte_addressable_store"

const cl_khr_device_enqueue_local_arg_types = 1

const CL_KHR_DEVICE_ENQUEUE_LOCAL_ARG_TYPES_EXTENSION_NAME = "cl_khr_device_enqueue_local_arg_types"

const cl_khr_expect_assume = 1

const CL_KHR_EXPECT_ASSUME_EXTENSION_NAME = "cl_khr_expect_assume"

const cl_khr_extended_async_copies = 1

const CL_KHR_EXTENDED_ASYNC_COPIES_EXTENSION_NAME = "cl_khr_extended_async_copies"

const cl_khr_extended_bit_ops = 1

const CL_KHR_EXTENDED_BIT_OPS_EXTENSION_NAME = "cl_khr_extended_bit_ops"

const cl_khr_global_int32_base_atomics = 1

const CL_KHR_GLOBAL_INT32_BASE_ATOMICS_EXTENSION_NAME = "cl_khr_global_int32_base_atomics"

const cl_khr_global_int32_extended_atomics = 1

const CL_KHR_GLOBAL_INT32_EXTENDED_ATOMICS_EXTENSION_NAME = "cl_khr_global_int32_extended_atomics"

const cl_khr_int64_base_atomics = 1

const CL_KHR_INT64_BASE_ATOMICS_EXTENSION_NAME = "cl_khr_int64_base_atomics"

const cl_khr_int64_extended_atomics = 1

const CL_KHR_INT64_EXTENDED_ATOMICS_EXTENSION_NAME = "cl_khr_int64_extended_atomics"

const cl_khr_kernel_clock = 1

const CL_KHR_KERNEL_CLOCK_EXTENSION_NAME = "cl_khr_kernel_clock"

const CL_DEVICE_KERNEL_CLOCK_CAPABILITIES_KHR = 0x1076

const CL_DEVICE_KERNEL_CLOCK_SCOPE_DEVICE_KHR = 1 << 0

const CL_DEVICE_KERNEL_CLOCK_SCOPE_WORK_GROUP_KHR = 1 << 1

const CL_DEVICE_KERNEL_CLOCK_SCOPE_SUB_GROUP_KHR = 1 << 2

const cl_khr_local_int32_base_atomics = 1

const CL_KHR_LOCAL_INT32_BASE_ATOMICS_EXTENSION_NAME = "cl_khr_local_int32_base_atomics"

const cl_khr_local_int32_extended_atomics = 1

const CL_KHR_LOCAL_INT32_EXTENDED_ATOMICS_EXTENSION_NAME = "cl_khr_local_int32_extended_atomics"

const cl_khr_mipmap_image_writes = 1

const CL_KHR_MIPMAP_IMAGE_WRITES_EXTENSION_NAME = "cl_khr_mipmap_image_writes"

const cl_khr_select_fprounding_mode = 1

const CL_KHR_SELECT_FPROUNDING_MODE_EXTENSION_NAME = "cl_khr_select_fprounding_mode"

const cl_khr_spirv_extended_debug_info = 1

const CL_KHR_SPIRV_EXTENDED_DEBUG_INFO_EXTENSION_NAME = "cl_khr_spirv_extended_debug_info"

const cl_khr_spirv_linkonce_odr = 1

const CL_KHR_SPIRV_LINKONCE_ODR_EXTENSION_NAME = "cl_khr_spirv_linkonce_odr"

const cl_khr_spirv_no_integer_wrap_decoration = 1

const CL_KHR_SPIRV_NO_INTEGER_WRAP_DECORATION_EXTENSION_NAME = "cl_khr_spirv_no_integer_wrap_decoration"

const cl_khr_srgb_image_writes = 1

const CL_KHR_SRGB_IMAGE_WRITES_EXTENSION_NAME = "cl_khr_srgb_image_writes"

const cl_khr_subgroup_ballot = 1

const CL_KHR_SUBGROUP_BALLOT_EXTENSION_NAME = "cl_khr_subgroup_ballot"

const cl_khr_subgroup_clustered_reduce = 1

const CL_KHR_SUBGROUP_CLUSTERED_REDUCE_EXTENSION_NAME = "cl_khr_subgroup_clustered_reduce"

const cl_khr_subgroup_extended_types = 1

const CL_KHR_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME = "cl_khr_subgroup_extended_types"

const cl_khr_subgroup_non_uniform_arithmetic = 1

const CL_KHR_SUBGROUP_NON_UNIFORM_ARITHMETIC_EXTENSION_NAME = "cl_khr_subgroup_non_uniform_arithmetic"

const cl_khr_subgroup_non_uniform_vote = 1

const CL_KHR_SUBGROUP_NON_UNIFORM_VOTE_EXTENSION_NAME = "cl_khr_subgroup_non_uniform_vote"

const cl_khr_subgroup_rotate = 1

const CL_KHR_SUBGROUP_ROTATE_EXTENSION_NAME = "cl_khr_subgroup_rotate"

const cl_khr_subgroup_shuffle = 1

const CL_KHR_SUBGROUP_SHUFFLE_EXTENSION_NAME = "cl_khr_subgroup_shuffle"

const cl_khr_subgroup_shuffle_relative = 1

const CL_KHR_SUBGROUP_SHUFFLE_RELATIVE_EXTENSION_NAME = "cl_khr_subgroup_shuffle_relative"

const cl_khr_work_group_uniform_arithmetic = 1

const CL_KHR_WORK_GROUP_UNIFORM_ARITHMETIC_EXTENSION_NAME = "cl_khr_work_group_uniform_arithmetic"

const cl_img_cancel_command = 1

const CL_IMG_CANCEL_COMMAND_EXTENSION_NAME = "cl_img_cancel_command"

const CL_CANCELLED_IMG = -1126
