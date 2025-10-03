# support for device-side exceptions

## exception type

struct KernelException <: Exception
    devs::Vector{cl.Device}
end

function Base.showerror(io::IO, err::KernelException)
    print(io, "KernelException: exception thrown during kernel execution on device(s) $(join(map(dev->dev.name, err.devs), ", "))")
end


## exception handling

const exception_infos = Dict{cl.Context, Union{Nothing, cl.AbstractPointerMemory}}()

# create a CPU/GPU exception flag for error signalling
function create_exceptions!(ctx::cl.Context, dev::cl.Device)
    mem = get!(exception_infos, ctx) do
        if cl.svm_capabilities(cl.device()).fine_grain_buffer
            cl.svm_alloc(sizeof(ExceptionInfo_st); fine_grained=true)
        elseif cl.usm_supported(dev) && cl.usm_capabilities(dev).host.access
            cl.host_alloc(sizeof(ExceptionInfo_st))
        else
            nothing
        end
    end
    if mem === nothing
        return convert(ExceptionInfo, C_NULL)
    end

    exception_info = convert(ExceptionInfo, mem)
    unsafe_store!(exception_info, ExceptionInfo_st())
    return exception_info
end

# check the exception flags on every API call
function check_exceptions()
    for (ctx, mem) in exception_infos
        mem === nothing && continue
        exception_info = convert(ExceptionInfo, mem)
        if exception_info.status != 0
            # restore the structure
            unsafe_store!(exception_info, ExceptionInfo_st())

            # throw host-side
            throw(KernelException(ctx.devices))
        end
    end
    return
end
