#----- OpenCL Image ----

const _img_chan_names = (CL_uint => Symbol)[CL_R => :CL_R, 
                                            CL_A => :CL_A,
                                            CL_RG => :CL_RG,
                                            CL_RGB => :CL_RGB, 
                                            CL_RGBA => :CL_RGBA,
                                            CL_BGRA => :CL_BGRA, 
                                            CL_ARGB => :CL_ARGB, 
                                            CL_INTENSITY => :CL_INTENSITY, 
                                            CL_LUMINANCE => :CL_LUMINANCE]

const _img_type_names = (CL_uint => Symbol)[CL_SNORM_INT8  => :CL_SNORM_INT8, 
                                            CL_SNORM_INT16 => :CL_SNORM_INT16,
                                            CL_UNORM_INT8  => :CL_UNORM_INT8, 
                                            CL_UNORM_INT16 => :CL_UNORM_INT16,
                                            CL_UNORM_INT24 => :CL_UNORM_INT24,
                                            CL_UNORM_SHORT_555  => :CL_UNORM_SHORT_555, 
                                            CL_UNORM_SHORT_565  => :CL_UNORM_SHORT_565,
                                            CL_UNORM_INT_101010 => :CL_UNORM_INT_101010,
                                            CL_SIGNED_INT8  => :CL_SIGNED_INT8, 
                                            CL_SIGNED_INT16 => :CL_SIGNED_INT16, 
                                            CL_SIGNED_INT32 => :CL_SIGNED_INT32,
                                            CL_UNSIGNED_INT8  => :CL_UNSIGNED_INT8, 
                                            CL_UNSIGNED_INT16 => :CL_UNSIGNED_INT16, 
                                            CL_UNSIGNED_INT32 => :CL_UNSIGNED_INT32,
                                            CL_FLOAT => :CL_FLOAT, 
                                            CL_HALF_FLOAT  => :CL_HALF_FLOAT, 
                                            CL_UNORM_INT24 => :CL_UNORM_INT24]

abstract CLImageFormat
immutable ImageFormat{C, T} <: CLImageFormat
end

Base.show{C, T}(io::IO, fmt::ImageFormat{C, T}) = begin
    ichan = _img_chan_names[C]
    itype = _img_type_names[T]
    print(io, "ImageFormat{$ichan, $itype}")
end

type Image{C, T} <: CLMemObject
    valid::Bool
    id::CL_mem

    function Image(mem_id::CL_mem, retain::Bool, len::Integer)
        @assert len > 0
        @assert mem_id != C_NULL
        if retain
            @check api.clRetainMemObject(mem_id)
        end
        img = new(true, mem_id)
        finalizer(img, mem_obj -> begin
            if !mem_obj.valid
                throw(CLMemoryError("attempted to double free OpenCL.Image $mem_obj"))
            end
            release!(mem_obj)
            mem_obj.valid   = false
            mem_obj.hostbuf = C_NULL
        end)
        return img
    end
end

function nchannels(fmt::CL_image_format)
    T = fmt.image_channel_order
    if T == CL_R
        return 1
    elseif T == CL_A
        return 1
    elseif T == CL_RG
        return 2
    elseif T == CL_RA
        return 2
    elseif T == CL_RGB
        return 3
    elseif T == CL_RGBA
        return 4
    elseif T == CL_BGRA
        return 4
    elseif T == CL_ARGB
        return 4
    elseif T == CL_INTENSITY
        return 1
    elseif T == CL_LUMINANCE
        return 1
    else
        # this should not happen
        error("unrecognized OpenCL channel order constant $T")
    end
end

function channel_size(fmt::CL_image_format)
    T = fmt.image_channel_data_type
    if T == CL_SNORM_INT8
        return 1
    elseif T == CL_SNORM_INT16
        return 2
    elseif T == CL_UNORM_INT8
        return 1
    elseif T == CL_UNORM_INT16
        return 2
    elseif T == CL_UNORM_INT24
        return 3
    elseif T == CL_UNORM_SHORT_555
        return 2
    elseif T == CL_UNORM_SHORT_565
        return 2
    elseif T == CL_UNORM_INT_101010
        return 4
    elseif T == CL_SIGNED_INT8
        return 1
    elseif T == CL_SIGNED_INT16
        return 2
    elseif T == CL_SIGNED_INT32
        return 4
    elseif T == CL_UNSIGNED_INT8
        return 1
    elseif T == CL_UNSIGNED_INT16
        return 2
    elseif T == CL_UNSIGNED_INT32
        return 4
    elseif T == CL_FLOAT
        return 4
    elseif T == CL_HALF_FLOAT
        return 2
    else
        # this should not happen
        error("unrecognized OpenCL channel data type constant $T")
    end
end

function image_format_item_size(fmt::CL_image_format)
    return nchannels(fmt) * channel_size(fmt)
end

function supported_image_formats(ctx::Context, 
                                 flags::CL_mem_flags,
                                 img_type::CL_mem_object_type)
    nformats = CL_uint[0]
    @check api.clGetSupportedImageFormats(ctx.id, flags, img_type, 
                                          0, C_NULL, nformats)
    if nformats[1] == 0
        return CL_image_format[]
    end
    formats  = Array(CL_image_format, nformats[1])
    @check api.clGetSupportedImageFormats(ctx.id, flags, img_type, 
                                          nformats[1], 
                                          isempty(formats) ? C_NULL : formats, C_NULL)
    return formats
end

function create_image{T}(ctx::Context, 
                         flags::CL_mem_flags,
                         fmt::CL_image_format,
                         shape::Dims,
                         pitches::Dims,
                         buffer::Buffer{T})
end

