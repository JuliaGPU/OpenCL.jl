# Error Codes
const CL_SUCCESS                                      =  0
const CL_DEVICE_NOT_FOUND                             = -1
const CL_DEVICE_NOT_AVAILABLE                         = -2
const CL_COMPILER_NOT_AVAILABLE                       = -3
const CL_MEM_OBJECT_ALLOCATION_FAILURE                = -4
const CL_OUT_OF_RESOURCES                             = -5
const CL_OUT_OF_HOST_MEMORY                           = -6
const CL_PROFILING_INFO_NOT_AVAILABLE                 = -7
const CL_MEM_COPY_OVERLAP                             = -8
const CL_IMAGE_FORMAT_MISMATCH                        = -9
const CL_IMAGE_FORMAT_NOT_SUPPORTED                   = -10
const CL_BUILD_PROGRAM_FAILURE                        = -11
const CL_MAP_FAILURE                                  = -12
const CL_MISALIGNED_SUB_BUFFER_OFFSET                 = -13
const CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST    = -14
const CL_COMPILE_PROGRAM_FAILURE                      = -15
const CL_LINKER_NOT_AVAILABLE                         = -15
const CL_LINK_PROGRAM_FAILURE                         = -17
const CL_DEVICE_PARTITION_FAILED                      = -18
const CL_KERNEL_ARG_INFO_NOT_AVAILABLE                = -19

const CL_INVALID_VALUE                                = -30
const CL_INVALID_DEVICE_TYPE                          = -31
const CL_INVALID_PLATFORM                             = -32
const CL_INVALID_DEVICE                               = -33
const CL_INVALID_CONTEXT                              = -34
const CL_INVALID_QUEUE_PROPERTIES                     = -35
const CL_INVALID_COMMAND_QUEUE                        = -36
const CL_INVALID_HOST_PTR                             = -37
const CL_INVALID_MEM_OBJECT                           = -38
const CL_INVALID_IMAGE_FORMAT_DESCRIPTOR              = -39
const CL_INVALID_IMAGE_SIZE                           = -40
const CL_INVALID_SAMPLER                              = -41
const CL_INVALID_BINARY                               = -42
const CL_INVALID_BUILD_OPTIONS                        = -43
const CL_INVALID_PROGRAM                              = -44
const CL_INVALID_PROGRAM_EXECUTABLE                   = -45
const CL_INVALID_KERNEL_NAME                          = -46
const CL_INVALID_KERNEL_DEFINITION                    = -47
const CL_INVALID_KERNEL                               = -48
const CL_INVALID_ARG_INDEX                            = -49
const CL_INVALID_ARG_VALUE                            = -50
const CL_INVALID_ARG_SIZE                             = -51
const CL_INVALID_KERNEL_ARGS                          = -52
const CL_INVALID_WORK_DIMENSION                       = -53
const CL_INVALID_WORK_GROUP_SIZE                      = -54
const CL_INVALID_WORK_ITEM_SIZE                       = -55
const CL_INVALID_GLOBAL_OFFSET                        = -56
const CL_INVALID_EVENT_WAIT_LIST                      = -57
const CL_INVALID_EVENT                                = -58
const CL_INVALID_OPERATION                            = -59
const CL_INVALID_GL_OBJECT                            = -60
const CL_INVALID_BUFFER_SIZE                          = -61
const CL_INVALID_MIP_LEVEL                            = -62
const CL_INVALID_GLOBAL_WORK_SIZE                     = -63
const CL_INVALID_PROPERTY                             = -64
const CL_INVALID_IMAGE_DESCRIPTOR                     = -65
const CL_INVALID_COMPILER_OPTIONS                     = -66
const CL_INVALID_LINKER_OPTIONS                       = -67
const CL_INVALID_DEVICE_PARTITION_COUNT               = -68

# OpenCL Version 
const CL_VERSION_1_0                                  = 1
const CL_VERSION_1_1                                  = 1
const CL_VERSION_1_2                                  = 1

# cl_bool
const CL_FALSE                                        = 0
const CL_TRUE                                         = 1
const CL_BLOCKING                                     = CL_TRUE
const CL_NON_BLOCKING                                 = CL_FALSE

# cl_platform_info
const CL_PLATFORM_PROFILE                             = 0x0900
const CL_PLATFORM_VERSION                             = 0x0901
const CL_PLATFORM_NAME                                = 0x0902
const CL_PLATFORM_VENDOR                              = 0x0903
const CL_PLATFORM_EXTENSIONS                          = 0x0904

# cl_device_type - bitfield 
const CL_DEVICE_TYPE_DEFAULT                          = (1 << 0)
const CL_DEVICE_TYPE_CPU                              = (1 << 1)
const CL_DEVICE_TYPE_GPU                              = (1 << 2)
const CL_DEVICE_TYPE_ACCELERATOR                      = (1 << 3)
const CL_DEVICE_TYPE_CUSTOM                           = (1 << 4)
const CL_DEVICE_TYPE_ALL                              = 0xFFFFFFFF

# cl_device_info 
const CL_DEVICE_TYPE                                  = 0x1000
const CL_DEVICE_VENDOR_ID                             = 0x1001
const CL_DEVICE_MAX_COMPUTE_UNITS                     = 0x1002
const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS              = 0x1003
const CL_DEVICE_MAX_WORK_GROUP_SIZE                   = 0x1004
const CL_DEVICE_MAX_WORK_ITEM_SIZES                   = 0x1005
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR           = 0x1006
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT          = 0x1007
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT            = 0x1008
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG           = 0x1009
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT          = 0x100A
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE         = 0x100B
const CL_DEVICE_MAX_CLOCK_FREQUENCY                   = 0x100C
const CL_DEVICE_ADDRESS_BITS                          = 0x100D
const CL_DEVICE_MAX_READ_IMAGE_ARGS                   = 0x100E
const CL_DEVICE_MAX_WRITE_IMAGE_ARGS                  = 0x100F
const CL_DEVICE_MAX_MEM_ALLOC_SIZE                    = 0x1010
const CL_DEVICE_IMAGE2D_MAX_WIDTH                     = 0x1011
const CL_DEVICE_IMAGE2D_MAX_HEIGHT                    = 0x1012
const CL_DEVICE_IMAGE3D_MAX_WIDTH                     = 0x1013
const CL_DEVICE_IMAGE3D_MAX_HEIGHT                    = 0x1014
const CL_DEVICE_IMAGE3D_MAX_DEPTH                     = 0x1015
const CL_DEVICE_IMAGE_SUPPORT                         = 0x1016
const CL_DEVICE_MAX_PARAMETER_SIZE                    = 0x1017
const CL_DEVICE_MAX_SAMPLERS                          = 0x1018
const CL_DEVICE_MEM_BASE_ADDR_ALIGN                   = 0x1019
const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE              = 0x101A
const CL_DEVICE_SINGLE_FP_CONFIG                      = 0x101B
const CL_DEVICE_GLOBAL_MEM_CACHE_TYPE                 = 0x101C
const CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE             = 0x101D
const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE                 = 0x101E
const CL_DEVICE_GLOBAL_MEM_SIZE                       = 0x101F
const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE              = 0x1020
const CL_DEVICE_MAX_CONSTANT_ARGS                     = 0x1021
const CL_DEVICE_LOCAL_MEM_TYPE                        = 0x1022
const CL_DEVICE_LOCAL_MEM_SIZE                        = 0x1023
const CL_DEVICE_ERROR_CORRECTION_SUPPORT              = 0x1024
const CL_DEVICE_PROFILING_TIMER_RESOLUTION            = 0x1025
const CL_DEVICE_ENDIAN_LITTLE                         = 0x1026
const CL_DEVICE_AVAILABLE                             = 0x1027
const CL_DEVICE_COMPILER_AVAILABLE                    = 0x1028
const CL_DEVICE_EXECUTION_CAPABILITIES                = 0x1029
const CL_DEVICE_QUEUE_PROPERTIES                      = 0x102A
const CL_DEVICE_NAME                                  = 0x102B
const CL_DEVICE_VENDOR                                = 0x102C
const CL_DRIVER_VERSION                               = 0x102D
const CL_DEVICE_PROFILE                               = 0x102E
const CL_DEVICE_VERSION                               = 0x102F
const CL_DEVICE_EXTENSIONS                            = 0x1030
const CL_DEVICE_PLATFORM                              = 0x1031
const CL_DEVICE_DOUBLE_FP_CONFIG                      = 0x1032

# 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG 
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF           = 0x1034
const CL_DEVICE_HOST_UNIFIED_MEMORY                   = 0x1035
const CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR              = 0x1036
const CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT             = 0x1037
const CL_DEVICE_NATIVE_VECTOR_WIDTH_INT               = 0x1038
const CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG              = 0x1039
const CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT             = 0x103A
const CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE            = 0x103B
const CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF              = 0x103C
const CL_DEVICE_OPENCL_C_VERSION                      = 0x103D
const CL_DEVICE_LINKER_AVAILABLE                      = 0x103E
const CL_DEVICE_BUILT_IN_KERNELS                      = 0x103F
const CL_DEVICE_IMAGE_MAX_BUFFER_SIZE                 = 0x1040
const CL_DEVICE_IMAGE_MAX_ARRAY_SIZE                  = 0x1041
const CL_DEVICE_PARENT_DEVICE                         = 0x1042
const CL_DEVICE_PARTITION_MAX_SUB_DEVICES             = 0x1043
const CL_DEVICE_PARTITION_PROPERTIES                  = 0x1044
const CL_DEVICE_PARTITION_AFFINITY_DOMAIN             = 0x1045
const CL_DEVICE_PARTITION_TYPE                        = 0x1046
const CL_DEVICE_REFERENCE_COUNT                       = 0x1047
const CL_DEVICE_PREFERRED_INTEROP_USER_SYNC           = 0x1048
const CL_DEVICE_PRINTF_BUFFER_SIZE                    = 0x1049
const CL_DEVICE_IMAGE_PITCH_ALIGNMENT                 = 0x104A
const CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT          = 0x104B

# cl_device_fp_config - bitfield 
const CL_FP_DENORM                                    = (1 << 0)
const CL_FP_INF_NAN                                   = (1 << 1)
const CL_FP_ROUND_TO_NEAREST                          = (1 << 2)
const CL_FP_ROUND_TO_ZERO                             = (1 << 3)
const CL_FP_ROUND_TO_INF                              = (1 << 4)
const CL_FP_FMA                                       = (1 << 5)
const CL_FP_SOFT_FLOAT                                = (1 << 6)
const CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT             = (1 << 7)

# cl_device_mem_cache_type
const CL_NONE                                         = 0x0
const CL_READ_ONLY_CACHE                              = 0x1
const CL_READ_WRITE_CACHE                             = 0x2

# cl_device_local_mem_type 
const CL_LOCAL                                        = 0x1
const CL_GLOBAL                                       = 0x2

# cl_device_exec_capabilities - bitfield 
const CL_EXEC_KERNEL                                  = (1 << 0)
const CL_EXEC_NATIVE_KERNEL                           = (1 << 1)

# cl_command_queue_properties - bitfield
const CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE          = (1 << 0)
const CL_QUEUE_PROFILING_ENABLE                       = (1 << 1)

# cl_context_info  
const CL_CONTEXT_REFERENCE_COUNT                      = 0x1080
const CL_CONTEXT_DEVICES                              = 0x1081
const CL_CONTEXT_PROPERTIES                           = 0x1082
const CL_CONTEXT_NUM_DEVICES                          = 0x1083

# cl_context_properties 
const CL_CONTEXT_PLATFORM                             = 0x1084
const CL_CONTEXT_INTEROP_USER_SYNC                    = 0x1085
    
# cl_device_partition_property 
const CL_DEVICE_PARTITION_EQUALLY                     = 0x1086
const CL_DEVICE_PARTITION_BY_COUNTS                   = 0x1087
const CL_DEVICE_PARTITION_BY_COUNTS_LIST_END          = 0x0
const CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN          = 0x1088
    
# cl_device_affinity_domain 
const CL_DEVICE_AFFINITY_DOMAIN_NUMA                  = (1 << 0)
const CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE              = (1 << 1)
const CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE              = (1 << 2)
const CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE              = (1 << 3)
const CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE              = (1 << 4)
const CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE    = (1 << 5)

# cl_command_queue_info
const CL_QUEUE_CONTEXT                                = 0x1090
const CL_QUEUE_DEVICE                                 = 0x1091
const CL_QUEUE_REFERENCE_COUNT                        = 0x1092
const CL_QUEUE_PROPERTIES                             = 0x1093

# cl_mem_flags - bitfield
const CL_MEM_READ_WRITE                               = (1 << 0)
const CL_MEM_WRITE_ONLY                               = (1 << 1)
const CL_MEM_READ_ONLY                                = (1 << 2)
const CL_MEM_USE_HOST_PTR                             = (1 << 3)
const CL_MEM_ALLOC_HOST_PTR                           = (1 << 4)
const CL_MEM_COPY_HOST_PTR                            = (1 << 5)
# //reserved                                            = (1 << 6)    
const CL_MEM_HOST_WRITE_ONLY                          = (1 << 7)
const CL_MEM_HOST_READ_ONLY                           = (1 << 8)
const CL_MEM_HOST_NO_ACCESS                           = (1 << 9)

# cl_mem_migration_flags - bitfield 
const CL_MIGRATE_MEM_OBJECT_HOST                      = (1 << 0)
const CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED         = (1 << 1)

# cl_channel_order 
const CL_R                                            = 0x10B0
const CL_A                                            = 0x10B1
const CL_RG                                           = 0x10B2
const CL_RA                                           = 0x10B3
const CL_RGB                                          = 0x10B4
const CL_RGBA                                         = 0x10B5
const CL_BGRA                                         = 0x10B6
const CL_ARGB                                         = 0x10B7
const CL_INTENSITY                                    = 0x10B8
const CL_LUMINANCE                                    = 0x10B9
const CL_Rx                                           = 0x10BA
const CL_RGx                                          = 0x10BB
const CL_RGBx                                         = 0x10BC
const CL_DEPTH                                        = 0x10BD
const CL_DEPTH_STENCIL                                = 0x10BE

# cl_channel_type
const CL_SNORM_INT8                                   = 0x10D0
const CL_SNORM_INT16                                  = 0x10D1
const CL_UNORM_INT8                                   = 0x10D2
const CL_UNORM_INT16                                  = 0x10D3
const CL_UNORM_SHORT_565                              = 0x10D4
const CL_UNORM_SHORT_555                              = 0x10D5
const CL_UNORM_INT_101010                             = 0x10D6
const CL_SIGNED_INT8                                  = 0x10D7
const CL_SIGNED_INT16                                 = 0x10D8
const CL_SIGNED_INT32                                 = 0x10D9
const CL_UNSIGNED_INT8                                = 0x10DA
const CL_UNSIGNED_INT16                               = 0x10DB
const CL_UNSIGNED_INT32                               = 0x10DC
const CL_HALF_FLOAT                                   = 0x10DD
const CL_FLOAT                                        = 0x10DE
const CL_UNORM_INT24                                  = 0x10DF

# cl_mem_object_type 
const CL_MEM_OBJECT_BUFFER                            = 0x10F0
const CL_MEM_OBJECT_IMAGE2D                           = 0x10F1
const CL_MEM_OBJECT_IMAGE3D                           = 0x10F2
const CL_MEM_OBJECT_IMAGE2D_ARRAY                     = 0x10F3
const CL_MEM_OBJECT_IMAGE1D                           = 0x10F4
const CL_MEM_OBJECT_IMAGE1D_ARRAY                     = 0x10F5
const CL_MEM_OBJECT_IMAGE1D_BUFFER                    = 0x10F6

# cl_mem_info 
const CL_MEM_TYPE                                     = 0x1100
const CL_MEM_FLAGS                                    = 0x1101
const CL_MEM_SIZE                                     = 0x1102
const CL_MEM_HOST_PTR                                 = 0x1103
const CL_MEM_MAP_COUNT                                = 0x1104
const CL_MEM_REFERENCE_COUNT                          = 0x1105
const CL_MEM_CONTEXT                                  = 0x1106
const CL_MEM_ASSOCIATED_MEMOBJECT                     = 0x1107
const CL_MEM_OFFSET                                   = 0x1108

# cl_image_info
const CL_IMAGE_FORMAT                                 = 0x1110
const CL_IMAGE_ELEMENT_SIZE                           = 0x1111
const CL_IMAGE_ROW_PITCH                              = 0x1112
const CL_IMAGE_SLICE_PITCH                            = 0x1113
const CL_IMAGE_WIDTH                                  = 0x1114
const CL_IMAGE_HEIGHT                                 = 0x1115
const CL_IMAGE_DEPTH                                  = 0x1116
const CL_IMAGE_ARRAY_SIZE                             = 0x1117
const CL_IMAGE_BUFFER                                 = 0x1118
const CL_IMAGE_NUM_MIP_LEVELS                         = 0x1119
const CL_IMAGE_NUM_SAMPLES                            = 0x111A

# cl_addressing_mode 
const CL_ADDRESS_NONE                                 = 0x1130
const CL_ADDRESS_CLAMP_TO_EDGE                        = 0x1131
const CL_ADDRESS_CLAMP                                = 0x1132
const CL_ADDRESS_REPEAT                               = 0x1133
const CL_ADDRESS_MIRRORED_REPEAT                      = 0x1134

# cl_filter_mode 
const CL_FILTER_NEAREST                               = 0x1140
const CL_FILTER_LINEAR                                = 0x1141

# cl_sampler_info 
const CL_SAMPLER_REFERENCE_COUNT                      = 0x1150
const CL_SAMPLER_CONTEXT                              = 0x1151
const CL_SAMPLER_NORMALIZED_COORDS                    = 0x1152
const CL_SAMPLER_ADDRESSING_MODE                      = 0x1153
const CL_SAMPLER_FILTER_MODE                          = 0x1154

# cl_map_flags - bitfield 
const CL_MAP_READ                                     = (1 << 0)
const CL_MAP_WRITE                                    = (1 << 1)
const CL_MAP_WRITE_INVALIDATE_REGION                  = (1 << 2)

# cl_program_info 
const CL_PROGRAM_REFERENCE_COUNT                      = 0x1160
const CL_PROGRAM_CONTEXT                              = 0x1161
const CL_PROGRAM_NUM_DEVICES                          = 0x1162
const CL_PROGRAM_DEVICES                              = 0x1163
const CL_PROGRAM_SOURCE                               = 0x1164
const CL_PROGRAM_BINARY_SIZES                         = 0x1165
const CL_PROGRAM_BINARIES                             = 0x1166
const CL_PROGRAM_NUM_KERNELS                          = 0x1167
const CL_PROGRAM_KERNEL_NAMES                         = 0x1168

# cl_program_build_info
const CL_PROGRAM_BUILD_STATUS                         = 0x1181
const CL_PROGRAM_BUILD_OPTIONS                        = 0x1182
const CL_PROGRAM_BUILD_LOG                            = 0x1183
const CL_PROGRAM_BINARY_TYPE                          = 0x1184
    
# cl_program_binary_type 
const CL_PROGRAM_BINARY_TYPE_NONE                     = 0x0
const CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT          = 0x1
const CL_PROGRAM_BINARY_TYPE_LIBRARY                  = 0x2
const CL_PROGRAM_BINARY_TYPE_EXECUTABLE               = 0x4

# cl_build_status
const CL_BUILD_SUCCESS                                = 0
const CL_BUILD_NONE                                   = -1
const CL_BUILD_ERROR                                  = -2
const CL_BUILD_IN_PROGRESS                            = -3

# cl_kernel_info 
const CL_KERNEL_FUNCTION_NAME                         = 0x1190
const CL_KERNEL_NUM_ARGS                              = 0x1191
const CL_KERNEL_REFERENCE_COUNT                       = 0x1192
const CL_KERNEL_CONTEXT                               = 0x1193
const CL_KERNEL_PROGRAM                               = 0x1194
const CL_KERNEL_ATTRIBUTES                            = 0x1195

# cl_kernel_arg_info 
const CL_KERNEL_ARG_ADDRESS_QUALIFIER                 = 0x1196
const CL_KERNEL_ARG_ACCESS_QUALIFIER                  = 0x1197
const CL_KERNEL_ARG_TYPE_NAME                         = 0x1198
const CL_KERNEL_ARG_TYPE_QUALIFIER                    = 0x1199
const CL_KERNEL_ARG_NAME                              = 0x119A

# cl_kernel_arg_address_qualifier 
const CL_KERNEL_ARG_ADDRESS_GLOBAL                    = 0x119B
const CL_KERNEL_ARG_ADDRESS_LOCAL                     = 0x119C
const CL_KERNEL_ARG_ADDRESS_CONSTANT                  = 0x119D
const CL_KERNEL_ARG_ADDRESS_PRIVATE                   = 0x119E

# cl_kernel_arg_access_qualifier 
const CL_KERNEL_ARG_ACCESS_READ_ONLY                  = 0x11A0
const CL_KERNEL_ARG_ACCESS_WRITE_ONLY                 = 0x11A1
const CL_KERNEL_ARG_ACCESS_READ_WRITE                 = 0x11A2
const CL_KERNEL_ARG_ACCESS_NONE                       = 0x11A3
    
# cl_kernel_arg_type_qualifer
const CL_KERNEL_ARG_TYPE_NONE                         = 0
const CL_KERNEL_ARG_TYPE_CONST                        = (1 << 0)
const CL_KERNEL_ARG_TYPE_RESTRICT                     = (1 << 1)
const CL_KERNEL_ARG_TYPE_VOLATILE                     = (1 << 2)

# cl_kernel_work_group_info 
const CL_KERNEL_WORK_GROUP_SIZE                       = 0x11B0
const CL_KERNEL_COMPILE_WORK_GROUP_SIZE               = 0x11B1
const CL_KERNEL_LOCAL_MEM_SIZE                        = 0x11B2
const CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE    = 0x11B3
const CL_KERNEL_PRIVATE_MEM_SIZE                      = 0x11B4
const CL_KERNEL_GLOBAL_WORK_SIZE                      = 0x11B5

# cl_event_info 
const CL_EVENT_COMMAND_QUEUE                          = 0x11D0
const CL_EVENT_COMMAND_TYPE                           = 0x11D1
const CL_EVENT_REFERENCE_COUNT                        = 0x11D2
const CL_EVENT_COMMAND_EXECUTION_STATUS               = 0x11D3
const CL_EVENT_CONTEXT                                = 0x11D4

# cl_command_type 
const CL_COMMAND_NDRANGE_KERNEL                       = 0x11F0
const CL_COMMAND_TASK                                 = 0x11F1
const CL_COMMAND_NATIVE_KERNEL                        = 0x11F2
const CL_COMMAND_READ_BUFFER                          = 0x11F3
const CL_COMMAND_WRITE_BUFFER                         = 0x11F4
const CL_COMMAND_COPY_BUFFER                          = 0x11F5
const CL_COMMAND_READ_IMAGE                           = 0x11F6
const CL_COMMAND_WRITE_IMAGE                          = 0x11F7
const CL_COMMAND_COPY_IMAGE                           = 0x11F8
const CL_COMMAND_COPY_IMAGE_TO_BUFFER                 = 0x11F9
const CL_COMMAND_COPY_BUFFER_TO_IMAGE                 = 0x11FA
const CL_COMMAND_MAP_BUFFER                           = 0x11FB
const CL_COMMAND_MAP_IMAGE                            = 0x11FC
const CL_COMMAND_UNMAP_MEM_OBJECT                     = 0x11FD
const CL_COMMAND_MARKER                               = 0x11FE
const CL_COMMAND_ACQUIRE_GL_OBJECTS                   = 0x11FF
const CL_COMMAND_RELEASE_GL_OBJECTS                   = 0x1200
const CL_COMMAND_READ_BUFFER_RECT                     = 0x1201
const CL_COMMAND_WRITE_BUFFER_RECT                    = 0x1202
const CL_COMMAND_COPY_BUFFER_RECT                     = 0x1203
const CL_COMMAND_USER                                 = 0x1204
const CL_COMMAND_BARRIER                              = 0x1205
const CL_COMMAND_MIGRATE_MEM_OBJECTS                  = 0x1206
const CL_COMMAND_FILL_BUFFER                          = 0x1207
const CL_COMMAND_FILL_IMAGE                           = 0x1208

# command execution status 
const CL_COMPLETE                                     = 0x0
const CL_RUNNING                                      = 0x1
const CL_SUBMITTED                                    = 0x2
const CL_QUEUED                                       = 0x3

# cl_buffer_create_type 
const CL_BUFFER_CREATE_TYPE_REGION                    = 0x1220

# cl_profiling_info  
const CL_PROFILING_COMMAND_QUEUED                     = 0x1280
const CL_PROFILING_COMMAND_SUBMIT                     = 0x1281
const CL_PROFILING_COMMAND_START                      = 0x1282
const CL_PROFILING_COMMAND_END                        = 0x1283
