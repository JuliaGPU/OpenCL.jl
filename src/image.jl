#----- OpenCL Image ----
export Red, Alpha, RG, RGB, RGBA, BGRA, ARGB, Intensity, Luminance,
       NormInt8, NormInt16, NormUint8, NormUint16, 
       NormUint24, NormUshort555, NormUshort565, NormInt101010,
       ImageFormat,
       Image

abstract CLImageChannel

immutable Red       <: CLImageChannel end 
immutable Alpha     <: CLImageChannel end
immutable RA        <: CLImageChannel end
immutable RG        <: CLImageChannel end
immutable RGB       <: CLImageChannel end
immutable RGBA      <: CLImageChannel end
immutable BGRA      <: CLImageChannel end
immutable ARGB      <: CLImageChannel end
immutable Intensity <: CLImageChannel end
immutable Luminance <: CLImageChannel end
immutable RX        <: CLImageChannel end 
immutable RGX       <: CLImageChannel end
immutable RGBX      <: CLImageChannel end
immutable Depth     <: CLImageChannel end
immutable DepthStencil <: CLImageChannel end

abstract NormImageType

immutable NormInt8   <: NormImageType end
immutable NormInt16  <: NormImageType end
immutable NormUint8  <: NormImageType end
immutable NormUint16 <: NormImageType end
immutable NormUint24 <: NormImageType end
immutable NormUshort555 <: NormImageType end
immutable NormUshort565 <: NormImageType end
immutable NormInt101010 <: NormImageType end

typealias CLImageType Union(NormImageType,
                            Int8, Uint8,
                            Int16, Uint16,
                            Int32, Uint32,
                            Float32, Float16)

function _swap_key_val{K,V}(d::Dict{K,V})
    return Dict{V, K}(collect(values(d)), 
                      collect(keys(d)))
end

const _img_chan_clconsts = (Type => CL_uint)[Red => CL_R, 
                                           Alpha => CL_A,
                                           RG    => CL_RG,
                                           RA    => CL_RA,
                                           RGB   => CL_RGB,
                                           RGBA  => CL_RGBA,
                                           BGRA  => CL_BGRA,
                                           ARGB  => CL_ARGB, 
                                           Intensity => CL_INTENSITY,
                                           Luminance => CL_LUMINANCE,
                                           RX => CL_Rx,
                                           RGX => CL_RGx,
                                           RGBX => CL_RGBx,
                                           Depth => CL_DEPTH, 
                                           DepthStencil => CL_DEPTH_STENCIL]

const _img_clconts_chan = _swap_key_val(_img_chan_clconsts)

const _img_type_clconsts = (Type => CL_uint)[NormInt8   => CL_SNORM_INT8, 
                                           NormInt16  => CL_SNORM_INT16,
                                           NormUint8  => CL_UNORM_INT8, 
                                           NormUint16 => CL_UNORM_INT16,
                                           NormUint24 => CL_UNORM_INT24,
                                           NormUshort555 => CL_UNORM_SHORT_555, 
                                           NormUshort565 => CL_UNORM_SHORT_565,
                                           NormInt101010 => CL_UNORM_INT_101010,
                                           Int8    => CL_SIGNED_INT8, 
                                           Int16   => CL_SIGNED_INT16, 
                                           Int32   => CL_SIGNED_INT32,
                                           Uint8   => CL_UNSIGNED_INT8, 
                                           Uint16  => CL_UNSIGNED_INT16, 
                                           Uint32  => CL_UNSIGNED_INT32,
                                           Float32 => CL_FLOAT, 
                                           Float16 => CL_HALF_FLOAT] 

const _img_clconts_type = _swap_key_val(_img_type_clconsts)

CL_image_format{C<:CLImageChannel, T<:CLImageType}(::Type{C}, ::Type{T}) = begin
    CL_image_format(_img_chan_clconsts[C], _img_type_clconsts[T])
end

type Image{C<:CLImageChannel, T<:CLImageType} <: CLMemObject
    valid::Bool
    id::CL_mem

    function Image(mem_id::CL_mem, retain::Bool)
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
            mem_obj.valid = false
            mem_obj.id = C_NULL
        end)
        return img
    end

    function Image(ctx::Context, mem_flag::Symbol, shape::Dims)
        dims = length(shape)
        itemsize = nchannels(C) * sizeof(T)
        flags = mem_flag === :r ? CL_MEM_READ_ONLY : CL_MEM_WRITE_ONLY
        fmt   = [CL_image_format(C, T)]
        err_code = Array(CL_int, 1)
        local mem_id::CL_mem
        if dims == 2
            width, height = shape
            @assert width >= 1 && height >= 1
            mem_id = api.clCreateImage2D(ctx.id, flags, fmt,
                                         width, height, 0,
                                         C_NULL, err_code)
        elseif dims == 3
            width, height, depth = shape
            @assert width >= 1 && height >= 1 && depth >= 1
            mem_id = api.clCreateImage3D(ctx.id, flags, fmt,
                                         width, height, depth,
                                         0, 0,
                                         C_NULL, err_code)
        else
            throw(ArgumentError("invalid dimension $dims"))
        end
        if err_code[1] != CL_SUCCESS
            throw(CLError(err_code[1]))
        end
        return Image{C,T}(mem_id, false)
    end

    function Image(ctx::Context, mem_flag::Symbol, arr::StridedArray)
        if !(mem_flag in (:rw, :r, :w))
            throw(ArgumentError("only one flag in {:rw, :r, :w} can be defined"))
        end
        if sizeof(arr) < 4#channel_size(C) 
            throw(ArgumentError("sizeof host array is less than image size"))
        end
        if size(arr)[end] != nchannels(C)
            throw(ArgumentError("first dimension must be equal to the number of channels"))
        end
        arr_ty   = eltype(arr)
        arr_ndim = ndims(arr)
        arr_size = size(arr)[2:end]
        img_nbytes = nchannels(C) * sizeof(T)
    end
end

#TODO: better error messages
function nchannels{T<:CLImageChannel}(::Type{T})
    if T === Red
        return 1
    elseif T === Alpha
        return 1
    elseif T === RG
        return 2
    elseif T === RGB
        return 3
    elseif T === RGBA
        return 4
    elseif T === BGRA
        return 4
    elseif T === ARGB
        return 4
    elseif T === Intensity
        return 1
    elseif T === Luminance
        return 1
    else
        return 1
        # this should not happen
        error("unrecognized OpenCL channel type $T")
    end
end

Base.sizeof{T<:CLImageType}(::Type{T}) = begin
    if T === NormInt8
        return 1
    elseif T === NormInt16
        return 2
    elseif T === NormUint8
        return 1
    elseif T === NormUint16
        return 2
    elseif T === NormUint24
        return 3
    elseif T === NormUshort555
        return 2
    elseif T === NormUshort565
        return 2
    elseif T === NormInt101010
        return 4
    elseif T === Int8
        return 1
    elseif T === Int16
        return 2
    elseif T === Int32
        return 4
    elseif T === Uint8
        return 1
    elseif T === Uint16
        return 2
    elseif T === Uint32
        return 4
    elseif T === Float32
        return 4
    elseif T === Float16
        return 2
    else
        # this should not happen
        error("unrecognized OpenCL channel data type $T")
    end
end

function supported_image_types(ctx::Context)
    t1 = supported_image_types(ctx, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D)
    t2 = supported_image_types(ctx, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE3D)
    return union(t1, t2)
end

function supported_image_types(ctx::Context, flag::Symbol, img_type::Symbol)
    local mf::CL_mem_flags
    if flag === :r
        mf = CL_MEM_READ_ONLY
    elseif flag === :w
        mf = CL_MEM_WRITE_ONLY
    elseif flag === :rw
        mf = CL_MEM_READ_WRITE
    else
        throw(ArgumentError("unrecognized flag :$flag"))
    end

    local ity::CL_mem_object_type
    if img_type === :image2d
        ity = CL_MEM_OBJECT_IMAGE2D
    elseif img_type === :image3d
        ity = CL_MEM_OBJECT_IMAGE3D
        #TODO: check context for ver >= 1.2 for rest of image object types
    else
        throw(ArgumentError("unrecognized img_type :$img_type"))
    end
    return supported_image_types(ctx, mf, ity)
end

function supported_image_types(ctx::Context, 
                               flags::CL_mem_flags,
                               img_type::CL_mem_object_type)
    nformats = CL_uint[0]
    @check api.clGetSupportedImageFormats(ctx.id, flags, img_type, 
                                          0, C_NULL, nformats)
    if nformats[1] == 0
        return Set{DataType}()
    end
    formats  = Array(CL_image_format, nformats[1])
    @check api.clGetSupportedImageFormats(ctx.id, flags, img_type, 
                                          nformats[1], 
                                          isempty(formats) ? C_NULL : pointer(formats), 
                                          C_NULL)
    img_formats = DataType[]
    for fmt in formats
       C = _img_clconts_chan[fmt.image_channel_order]
       T = _img_clconts_type[fmt.image_channel_data_type]
       push!(img_formats, Image{C,T})
    end
    return Set(img_formats)
end
