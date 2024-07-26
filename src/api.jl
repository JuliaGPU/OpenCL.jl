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

function parse_version(version_string)
    mg = match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
    if mg === nothing
        error("Non conforming version string: $(ver)")
    end
    return VersionNumber(parse(Int, mg.captures[1]),
                                 parse(Int, mg.captures[2]))
end
