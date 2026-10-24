# In-suite check for the intermittent KernelAbstractions `groupreduce` failure on macOS PoCL
# (`Float32, max` on the OpenCL C target). Included by test/kernelabstractions.jl, so that it
# runs in the worker and conditions in which KernelAbstractions' own test failed: before and
# after KernelAbstractions' test suite. Failures are printed with a diagnosis, and appended to
# `GROUPREDUCE_DEBUG_DIR/failures.log`; the kernels' OpenCL C and SPIR-V are dumped to
# `GROUPREDUCE_DEBUG_DIR/dumps`. Kernel names start with `groupreduce_dbg`, which identifies
# their entries in PoCL's kernel cache.

using KernelAbstractions, Printf

@kernel function groupreduce_dbg_static!(out, @Const(x), op, neutral)
    i = @index(Global, Linear)
    res = @groupreduce(op, x[i], neutral)
    out[i] = res
end

@kernel function groupreduce_dbg_bound!(out, @Const(x), op, neutral, ::Val{N}) where {N}
    i = @index(Global, Linear)
    res = @groupreduce(op, x[i], neutral; groupsize = N)
    out[i] = res
end

groupreduce_dbg_max(a, b) = max(a, b)  # not `max`: integers skip the native reduction

const GROUPREDUCE_DEBUG_DIR = mkpath(get(ENV, "GROUPREDUCE_DEBUG_DIR", joinpath(tempdir(), "groupreduce-debug")))

function groupreduce_dbg_report(msg)
    println(msg)
    open(io -> println(io, msg), joinpath(GROUPREDUCE_DEBUG_DIR, "failures.log"), "a")
end

function groupreduce_dbg_ranges(lanes)
    isempty(lanes) && return "none"
    parts = String[]
    start = prev = lanes[1]
    for l in lanes[2:end]
        if l != prev + 1
            push!(parts, start == prev ? "$start" : "$start:$prev")
            start = l
        end
        prev = l
    end
    push!(parts, start == prev ? "$start" : "$start:$prev")
    return join(parts, ",")
end

# compare with the reduction of every work-group, and describe the wrong work-groups: the
# lanes with values the result doesn't account for, and whether all lanes agree
function groupreduce_dbg_compare(got, x, gs, op, what)
    n = length(x)
    expected = similar(x)
    for g in 1:gs:n
        r = g:min(g + gs - 1, n)
        expected[r] .= reduce(op, x[r])
    end
    got == expected && return true

    bad = [g for g in 1:gs:n if got[g:min(g + gs - 1, n)] != expected[g:min(g + gs - 1, n)]]
    groupreduce_dbg_report("GROUPREDUCE FAILURE in $what, pid $(getpid()), $(length(bad))/$(cld(n, gs)) work-groups wrong")
    for g in first(bad, 4)
        r = g:min(g + gs - 1, n)
        vals = unique(got[r])
        v = minimum(vals)
        groupreduce_dbg_report("  elements $(first(r)):$(last(r)): got $(length(vals) == 1 ? v : "disagreeing $(first(vals, 8))"), expected $(expected[first(r)]); " *
                               "lanes above the result: $(groupreduce_dbg_ranges(findall(>(v), x[r]) .- 1)); result $(v in x[r] ? "is" : "is NOT") in the work-group")
        groupreduce_dbg_report("    got:   $(got[r][1:min(end, 64)])")
        groupreduce_dbg_report("    input: $(x[r][1:min(end, 64)])")
    end
    return false
end

# the launches of KernelAbstractions' test, for `Float32` `max` and a control that takes the
# same shuffle path for integers
function groupreduce_check(when)
    backend = OpenCL.program_backend()
    dumps = mkpath(joinpath(GROUPREDUCE_DEBUG_DIR, "dumps"))
    old = (get(ENV, "JULIA_OPENCL_DUMP_DIR", nothing), get(ENV, "BUILDKITE", nothing))
    ENV["JULIA_OPENCL_DUMP_DIR"] = dumps
    ENV["BUILDKITE"] = "false"  # don't upload every dump; the job uploads them on failure
    ok = true
    try
        b = OpenCLBackend()
        for (T, op) in ((Float32, max), (Int32, groupreduce_dbg_max))
            for (gs, n) in ((64, 64), (64, 256), (32, 100), (256, 1000), (7, 23), (1, 3))
                x = T.(rand(1:100, n))
                out = CLArray(fill(zero(T), n))
                groupreduce_dbg_static!(b, gs)(out, CLArray(x), op, typemin(T); ndrange = n)
                ok &= groupreduce_dbg_compare(Array(out), x, gs, op, "static $T $(nameof(op)) gs=$gs n=$n, $backend, $when")

                fill!(out, zero(T))
                groupreduce_dbg_bound!(b)(out, CLArray(x), op, typemin(T), Val(256); ndrange = n, workgroupsize = gs)
                ok &= groupreduce_dbg_compare(Array(out), x, gs, op, "bound $T $(nameof(op)) gs=$gs n=$n, $backend, $when")
            end
        end
    finally
        for (k, v) in zip(("JULIA_OPENCL_DUMP_DIR", "BUILDKITE"), old)
            v === nothing ? delete!(ENV, k) : (ENV[k] = v)
        end
    end
    return ok
end
