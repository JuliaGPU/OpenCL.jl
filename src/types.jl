#--- TypeAliases ---

# Opaque types
typealias CL_platform_id        Ptr{Void}
typealias CL_device_id          Ptr{Void}
typealias CL_context            Ptr{Void}
typealias CL_command_queue      Ptr{Void}
typealias CL_mem                Ptr{Void}
typealias CL_program            Ptr{Void}
typealias CL_kernel             Ptr{Void}
typealias CL_event              Ptr{Void}
typealias CL_sampler            Ptr{Void}

# Scalar types
typealias CL_char   Int8
typealias CL_uchar  Uint8
typealias CL_short  Int16
typealias CL_ushort Uint16
typealias CL_int    Int32
typealias CL_uint   Uint32
typealias CL_long   Int64
typealias CL_ulong  Uint64

typealias CL_half   Uint16
typealias CL_float  Float32
typealias CL_double Float64

typealias CL_bool                       CL_uint
typealias CL_bitfield                   CL_ulong
typealias CL_device_type                CL_bitfield
typealias CL_platform_info              CL_uint
typealias CL_device_info                CL_uint
typealias CL_device_fp_config           CL_bitfield
typealias CL_device_mem_cache_type      CL_uint
typealias CL_device_exec_capabilities   CL_bitfield
typealias CL_command_queue_properties   CL_bitfield

typealias CL_context_properties         Ptr{Cint}
typealias CL_context_info               CL_uint
typealias CL_command_queue_info         CL_uint
typealias CL_channel_order              CL_uint
typealias CL_channel_type               CL_uint
typealias CL_mem_flags                  CL_bitfield
typealias CL_mem_object_type            CL_uint
typealias CL_mem_info                   CL_uint
typealias CL_image_info                 CL_uint
typealias CL_buffer_create_type         CL_uint
typealias CL_addressing_mode            CL_uint
typealias CL_filter_mode                CL_uint
typealias CL_sampler_info               CL_uint
typealias CL_map_flags                  CL_bitfield
typealias CL_program_info               CL_uint
typealias CL_program_build_info         CL_uint
typealias CL_build_status               CL_int
typealias CL_kernel_info                CL_uint
typealias CL_kernel_work_group_info     CL_uint
typealias CL_event_info                 CL_uint
typealias CL_command_type               CL_uint
typealias CL_profiling_info             CL_uint

#--- Image Types ---

immutable ImageFormat
    image_channel_order::CL_channel_order
    image_channel_data_type::CL_channel_type 
end

type ImageDesc
    image_type::CL_mem_object_type
    image_width::Csize_t
    image_depth::Csize_t
    image_array_size::Csize_t
    image_row_pitch::Csize_t
    image_slice_pitch::Csize_t
    num_mip_levels::CL_uint
    num_samples::CL_uint
    buffer::CL_mem
end

immutable BufferRegion
    origin::Csize_t
    size::Csize_t
end


#--- Conversion Functions ---

cL_char(x)   = int8(x)
cl_uchar(x)  = uint8(x)
cl_short(x)  = int16(x)
cl_ushort(x) = uint16(x)
cl_int(x)    = int32(x)
cl_uint(x)   = uint32(x)
cl_long(x)   = int64(x)
cl_ulong(x)  = uint64(x)

cl_half(x)   = uint16(x)
cl_float(x)  = float32(x) 
cl_double(x) = float64(x)

cl_bool(x)     = cl_uint(x)
cl_bitfield(x) = cl_ulong(x) 


