import OpenCL_jll

const libopencl = OpenCL_jll.libopencl

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

macro ext_ccall(ex)
    # decode the expression
    @assert Meta.isexpr(ex, :(::))
    call, ret = ex.args
    @assert Meta.isexpr(call, :call)
    target, argexprs... = call.args
    @assert Meta.isexpr(target, :(.))
    _, fn = target.args

    @gensym fptr
    esc(quote
        $fptr = $clGetExtensionFunctionAddressForPlatform(platform(), $fn)
        @ccall $(Expr(:($), fptr))($(argexprs...))::$ret
    end)
end

include("libopencl.jl")

@static if Sys.iswindows()
    # Windows type aliases
    const BOOL = Int32
    const DWORD = UInt32
    const PDWORD = Ptr{DWORD}
    const HANDLE = Ptr{Cvoid}
    const PHANDLE = Ptr{HANDLE}
    const BYTE = UInt8
    const PBYTE = Ptr{BYTE}
    const PVOID = Ptr{Cvoid}
    const PSID = PVOID
    const PSID_AND_ATTRIBUTES = PVOID
    struct SID_AND_ATTRIBUTES
        Sid::PSID
        Attributes::DWORD
    end
    struct TOKEN_MANDATORY_LABEL
        Label::SID_AND_ATTRIBUTES
    end

    # Windows constants
    const TOKEN_QUERY = DWORD(0x0008)
    const TOKEN_QUERY_SOURCE = DWORD(0x0010)
    const SECURITY_MAX_SID_SIZE = 68
    const SECURITY_MANDATORY_MEDIUM_RID = DWORD(0x2000)
    const TokenIntegrityLevel = 25  # TOKEN_INFORMATION_CLASS enum value

    const kernel32 = "kernel32.dll"
    GetCurrentProcess() = @ccall kernel32.GetCurrentProcess()::HANDLE
    CloseHandle(hObject) = @ccall kernel32.CloseHandle(hObject::HANDLE)::BOOL

    const advapi32 = "advapi32.dll"
    OpenProcessToken(ProcessHandle, DesiredAccess, TokenHandle) =
        @ccall advapi32.OpenProcessToken(ProcessHandle::HANDLE, DesiredAccess::DWORD, TokenHandle::PHANDLE)::BOOL
    GetTokenInformation(TokenHandle, TokenInformationClass, TokenInformation, TokenInformationLength, ReturnLength) =
        @ccall advapi32.GetTokenInformation(TokenHandle::HANDLE, TokenInformationClass::Int32, TokenInformation::PBYTE, TokenInformationLength::DWORD, ReturnLength::PDWORD)::BOOL
    GetSidSubAuthorityCount(pSid) =
        @ccall advapi32.GetSidSubAuthorityCount(pSid::PSID)::PBYTE
    GetSidSubAuthority(pSid, nSubAuthority) =
        @ccall advapi32.GetSidSubAuthority(pSid::PSID, nSubAuthority::DWORD)::PDWORD

    # mimics `khrIcd_IsHighIntegrityLevel`, which determines if we can specify
    # drivers through environment variables
    function is_high_integrity_level()::Bool
        is_high_integrity = false

        h_token = Ref{HANDLE}()
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_QUERY_SOURCE, h_token) != 0
            try
                # Maximum possible size of SID_AND_ATTRIBUTES is maximum size of a SID + size of attributes DWORD.
                mandatory_label_buffer = zeros(BYTE, SECURITY_MAX_SID_SIZE + sizeof(DWORD))
                buffer_size = Ref{DWORD}()
                if GetTokenInformation(h_token[], TokenIntegrityLevel, mandatory_label_buffer, length(mandatory_label_buffer), buffer_size) != 0
                    mandatory_label = unsafe_load(Ptr{TOKEN_MANDATORY_LABEL}(pointer(mandatory_label_buffer)))
                    sub_auth_count = unsafe_load(GetSidSubAuthorityCount(mandatory_label.Label.Sid))
                    integrity_level = unsafe_load(GetSidSubAuthority(mandatory_label.Label.Sid, sub_auth_count - 1))

                    return integrity_level > SECURITY_MANDATORY_MEDIUM_RID
                end
            finally
                CloseHandle(h_token[])
            end
        end

        return is_high_integrity
    end
end

# lazy initialization
const initialized = Ref{Bool}(false)
@noinline function initialize()
    initialized[] = true

    @static if Sys.iswindows()
        if is_high_integrity_level()
            @warn """Running at high integrity level, preventing OpenCL.jl from loading drivers from JLLs.

                     Only system drivers will be available. To enable JLL drivers, do not run Julia as an administrator."""
        end
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

        if num_platforms[] == 0 && isempty(OpenCL_jll.drivers)
            @error """No OpenCL drivers available, either system-wide or provided by a JLL.

                      Please install a system-wide OpenCL driver, or load one together with OpenCL.jl,
                      e.g., by doing `using OpenCL, pocl_jll`."""
        end
    end
end

function __init__()
    if !OpenCL_jll.is_available()
        @error "OpenCL_jll is not available for your platform, OpenCL.jl. will not work."
    end

    # ensure that operations executed by the REPL back-end finish before returning,
    # because displaying values happens on a different task
    if isdefined(Base, :active_repl_backend) && !isnothing(Base.active_repl_backend)
        push!(Base.active_repl_backend.ast_transforms, synchronize_opencl_tasks)
    end
end

function synchronize_opencl_tasks(ex)
    quote
        try
            $(ex)
        finally
            if haskey($task_local_storage(), :CLDevice)
                $device_synchronize()
            end
        end
    end
end
