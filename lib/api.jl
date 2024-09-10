const libopencl = if Sys.isapple()
    "/System/Library/Frameworks/OpenCL.framework/OpenCL"
else
    import OpenCL_jll
    OpenCL_jll.libopencl
end

"""
    @checked function foo(...)
        rv = ...
        return rv
    end

Macro for wrapping a function definition returning a status code. Two versions of the
function will be generated: `foo`, with the function body wrapped by an invocation of the
`check` function (to be implemented by the caller of this macro), and `unchecked_foo` where no
such invocation is present and the status code is returned to the caller.
"""
macro checked(ex)
    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)

    # we need to detect the first API call, so add an initialization check
    body = quote
        if !initialized[]
            initialize()
        end
        $body
    end

    # generate a "safe" version that performs a check
    safe_body = quote
        check() do
            $body
        end
    end
    safe_sig = Expr(:call, sig.args[1], sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unchecked" version that returns the error code instead
    unchecked_sig = Expr(:call, Symbol("unchecked_", sig.args[1]), sig.args[2:end]...)
    unchecked_def = Expr(:function, unchecked_sig, body)

    return esc(:($safe_def, $unchecked_def))
end

function retry_reclaim(f, isfailed)
    ret = f()

    # slow path, incrementally reclaiming more memory until we succeed
    if isfailed(ret)
        phase = 1
        while true
            if phase == 1
                GC.gc(false)
            elseif phase == 2
                GC.gc(true)
            else
                break
            end
            phase += 1

            ret = f()
            isfailed(ret) || break
        end
    end

    ret
end

include("../lib/libopencl.jl")

# lazy initialization
const initialized = Ref{Bool}(false)
@noinline function initialize()
    initialized[] = true

    Sys.isapple() && return

    if isempty(OpenCL_jll.drivers)
        @warn """No OpenCL driver JLLs were detected at the time of the first call into OpenCL.jl.
                 Only system drivers will be available."""
        return
    end

    ocd_filenames = join(OpenCL_jll.drivers, ':')
    if haskey(ENV, "OCL_ICD_FILENAMES")
        ocd_filenames *= ":" * ENV["OCL_ICD_FILENAMES"]
    end

    withenv("OCL_ICD_FILENAMES"=>ocd_filenames) do
        num_platforms = Ref{Cuint}()
        @ccall libopencl.clGetPlatformIDs(
            0::cl_uint, C_NULL::Ptr{cl_platform_id},
            num_platforms::Ptr{cl_uint})::cl_int
    end
end

function __init__()
    if Sys.isapple()
        @warn "on macOS, OpenCL.jl uses the system OpenCL framework, which is deprecated."
    elseif !OpenCL_jll.is_available()
        @error "OpenCL_jll is not available for your platform, OpenCL.jl. will not work."
    end
end

const _versionDict = Dict{Ptr, VersionNumber}()
_deletecached!(obj::CLObject) = delete!(_versionDict, pointer(obj))
