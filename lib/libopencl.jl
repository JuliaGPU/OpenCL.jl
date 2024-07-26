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

const intptr_t = Clong

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
