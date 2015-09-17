const _cl_error_codes = @compat Dict{Int, Symbol}(
     +0 => :CL_SUCCESS,
     -1 => :CL_DEVICE_NOT_FOUND,
     -2 => :CL_DEVICE_NOT_AVAILABLE,
     -3 => :CL_COMPILER_NOT_AVAILABLE,
     -4 => :CL_MEM_OBJECT_ALLOCATION_FAILURE,
     -5 => :CL_OUT_OF_RESOURCES,
     -6 => :CL_OUT_OF_HOST_MEMORY,
     -7 => :CL_PROFILING_INFO_NOT_AVAILABLE,
     -8 => :CL_MEM_COPY_OVERLAP,
     -9 => :CL_IMAGE_FORMAT_MISMATCH,
    -10 => :CL_IMAGE_FORMAT_NOT_SUPPORTED,
    -11 => :CL_BUILD_PROGRAM_FAILURE,
    -12 => :CL_MAP_FAILURE,
    -13 => :CL_MISALIGNED_SUB_BUFFER_OFFSET,
    -14 => :CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
    -15 => :CL_COMPILE_PROGRAM_FAILURE,
    -15 => :CL_LINKER_NOT_AVAILABLE,
    -17 => :CL_LINK_PROGRAM_FAILURE,
    -18 => :CL_DEVICE_PARTITION_FAILED,
    -19 => :CL_KERNEL_ARG_INFO_NOT_AVAILABLE,

    -30 => :CL_INVALID_VALUE,
    -31 => :CL_INVALID_DEVICE_TYPE,
    -32 => :CL_INVALID_PLATFORM,
    -33 => :CL_INVALID_DEVICE,
    -34 => :CL_INVALID_CONTEXT,
    -35 => :CL_INVALID_QUEUE_PROPERTIES,
    -36 => :CL_INVALID_COMMAND_QUEUE,
    -37 => :CL_INVALID_HOST_PTR,
    -38 => :CL_INVALID_MEM_OBJECT,
    -39 => :CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
    -40 => :CL_INVALID_IMAGE_SIZE,
    -41 => :CL_INVALID_SAMPLER,
    -42 => :CL_INVALID_BINARY,
    -43 => :CL_INVALID_BUILD_OPTIONS,
    -44 => :CL_INVALID_PROGRAM,
    -45 => :CL_INVALID_PROGRAM_EXECUTABLE,
    -46 => :CL_INVALID_KERNEL_NAME,
    -47 => :CL_INVALID_KERNEL_DEFINITION,
    -48 => :CL_INVALID_KERNEL,
    -49 => :CL_INVALID_ARG_INDEX,
    -50 => :CL_INVALID_ARG_VALUE,
    -51 => :CL_INVALID_ARG_SIZE,
    -52 => :CL_INVALID_KERNEL_ARGS,
    -53 => :CL_INVALID_WORK_DIMENSION,
    -54 => :CL_INVALID_WORK_GROUP_SIZE,
    -55 => :CL_INVALID_WORK_ITEM_SIZE,
    -56 => :CL_INVALID_GLOBAL_OFFSET,
    -57 => :CL_INVALID_EVENT_WAIT_LIST,
    -58 => :CL_INVALID_EVENT,
    -59 => :CL_INVALID_OPERATION,
    -60 => :CL_INVALID_GL_OBJECT,
    -61 => :CL_INVALID_BUFFER_SIZE,
    -62 => :CL_INVALID_MIP_LEVEL,
    -63 => :CL_INVALID_GLOBAL_WORK_SIZE,
    -64 => :CL_INVALID_PROPERTY,
    -65 => :CL_INVALID_IMAGE_DESCRIPTOR,
    -66 => :CL_INVALID_COMPILER_OPTIONS,
    -67 => :CL_INVALID_LINKER_OPTIONS,
    -68 => :CL_INVALID_DEVICE_PARTITION_COUNT,
    -69 => :CL_INVALID_PIPE_SIZE,
    -70 => :CL_INVALID_DEVICE_QUEUE,

    -1000 => :CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR,
    -1001 => :CL_PLATFORM_NOT_FOUND_KHR,
    -1002 => :CL_INVALID_D3D10_DEVICE_KHR,
    -1003 => :CL_INVALID_D3D10_RESOURCE_KHR,
    -1004 => :CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR,
    -1005 => :CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR,
    -1006 => :CL_INVALID_D3D11_DEVICE_KHR,
    -1007 => :CL_INVALID_D3D11_RESOURCE_KHR,
    -1008 => :CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR,
    -1009 => :CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR,
    -1010 => :CL_INVALID_DX9_MEDIA_ADAPTER_KHR,
    -1011 => :CL_INVALID_DX9_MEDIA_SURFACE_KHR,
    -1012 => :CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR,
    -1013 => :CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR,

    -1057 => :CL_DEVICE_PARTITION_FAILED_EXT,
    -1058 => :CL_INVALID_PARTITION_COUNT_EXT,
    -1059 => :CL_INVALID_PARTITION_NAME_EXT,

    -1092 => :CL_EGL_RESOURCE_NOT_ACQUIRED_KHR,
    -1093 => :CL_INVALID_EGL_OBJECT_KHR,
)

const _cl_err_desc = @compat Dict{Integer, AbstractString}(
    CL_INVALID_CONTEXT =>
    "Context is not a valid context.",

    CL_INVALID_BUFFER_SIZE =>
    "Buffer size is 0",

    CL_INVALID_EVENT =>
    "Event objects specified in event_list are not valid event objects",

    CL_INVALID_HOST_PTR =>
    string("If host_ptr is NULL and CL_MEM_USE_HOST_PTR or ",
           "CL_MEM_COPY_HOST_PTR are set in flags or if host_ptr is not NULL but ",
           "CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags."),

    CL_MEM_OBJECT_ALLOCATION_FAILURE =>
    "Failure to allocate memory for buffer object.",

    CL_OUT_OF_RESOURCES =>
    "Failure to allocate resources required by the OpenCL implementation on the device.",

    CL_OUT_OF_HOST_MEMORY =>
    "Failure to allocate resources required by the OpenCL implementation on the host",

    CL_INVALID_PROGRAM =>
    "Program is not a valid program object.",

    CL_INVALID_VALUE =>
    "CL_INVALID_VALUE: this one should have been caught by julia!",

    CL_INVALID_DEVICE =>
    "OpenCL devices listed in device_list are not in the list of devices associated with program.",

    CL_INVALID_BINARY =>
    string("program is created with clCreateWithProgramBinary and devices listed in ",
           "device_list do not have a valid program binary loaded."),

    CL_INVALID_BUILD_OPTIONS =>
    "The build options specified by options are invalid.",

    CL_INVALID_OPERATION =>
    string("The build of a program executable for any of the devices listed in device_list by a ",
           "previous call to clBuildProgram for program has not  completed."),

    CL_COMPILER_NOT_AVAILABLE =>
    "Program is created with clCreateProgramWithSource and a compiler is not available",

    CL_BUILD_PROGRAM_FAILURE =>
    string("Failure to build the program executable. ",
           "This error will be returned if clBuildProgram ",
           "does not return until the build has completed"),

    CL_INVALID_OPERATION =>
    "There are kernel objects attached to program.",

    CL_OUT_OF_HOST_MEMORY =>
    "if there is a failure to allocate resources required by the OpenCL implementation on the host.",

    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST =>
    "The execution status of any of the events in event_list is a negative integer value",

    CL_INVALID_PROGRAM_EXECUTABLE =>
    "there is no successfully built executable for program",

    CL_INVALID_KERNEL_NAME =>
    "kernel_name is not found in program.",

    CL_INVALID_KERNEL_DEFINITION =>
    string("The function definition for __kernel  function ",
           "given by kernel_name such as the number of arguments, the argument types are not the ",
           "same for all devices for which the program executable has been built"),

    CL_PROFILING_INFO_NOT_AVAILABLE =>
    string("The CL_QUEUE_PROFILING_ENABLE flag ",
           "is not set for the command-queue, if the execution status of the command identified by ",
           "event is not CL_COMPLETE or if event is a user event objec"),
)

immutable CLMemoryError <: Exception
    msg::AbstractString
end

Base.show(io::IO, err::CLMemoryError) = Base.print(io, "OpenCL.MemObject: $(err.msg)")

immutable OpenCLException <: Exception
    msg::AbstractString
end

Base.show(io::IO, err::OpenCLException) = Base.print(io, "OpenCL Error: $(err.msg)")

immutable CLError <: Exception
    code::CL_int
    desc::Symbol

    function CLError(c::Integer)
        @compat new(c, _cl_error_codes[Int(c)])
    end
end

Base.show(io::IO, err::CLError) =
        Base.print(io, "CLError(code=$(err.code), $(err.desc))")

function error_description(err::CLError)
    try
        _cl_error_descriptions[err.code]
    catch
        return "no description for error $(err.desc)"
    end
end
