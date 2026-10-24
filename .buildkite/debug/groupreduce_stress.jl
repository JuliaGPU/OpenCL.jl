# Stress test for the intermittent KernelAbstractions `groupreduce` failure on macOS PoCL
# (`Float32, max` on the OpenCL C target, see #581 / Buildkite builds 1524 and 1539).
#
#   julia --project=test groupreduce_stress.jl DRIVER PHASES SECONDS OUTDIR
#
# DRIVER is `pocl` or `pocl_next`; PHASES is a comma-separated list of
#  - `warm`:    compile once, then launch the kernels over and over
#  - `cpuload`: `warm`, with 2 busy processes per core
#  - `gc`:      `warm`, with a task that allocates garbage and runs the GC in between launches
#  - `cold`:    compile fresh copies of the kernels every round and launch each once per
#               configuration, as the KernelAbstractions test suite does
#  - `sibling`: `cold`, while a second Julia process runs `cold` too
# Every failure is logged to OUTDIR/failures.log with a diagnosis of which lanes the
# reduction left out. Kernels are dumped (OpenCL C and SPIR-V) to OUTDIR/dumps, and PoCL's
# kernel cache, with the binaries that actually ran, is kept in POCL_CACHE_DIR.

const DRIVER = ARGS[1]
const PHASES = split(ARGS[2], ',')
const SECONDS = parse(Float64, ARGS[3])
const OUTDIR = mkpath(ARGS[4])

# dump every kernel, but don't upload each one: the job uploads OUTDIR at the end
ENV["JULIA_OPENCL_DUMP_DIR"] = mkpath(joinpath(OUTDIR, "dumps"))
ENV["BUILDKITE"] = "false"

using OpenCL, KernelAbstractions, Printf, Random, InteractiveUtils
import KernelInterface as KI
if DRIVER == "pocl_next"
    using pocl_next_jll
else
    using pocl_jll
end

const LOG = open(joinpath(OUTDIR, "failures.log"), "a")
function report(msg)
    println(msg)
    println(LOG, msg)
    flush(LOG)
    flush(stdout)
end

const T0 = time()
stamp() = @sprintf("%8.1fs", time() - T0)

const CONFIGS = ((64, 64), (64, 256), (32, 100), (256, 1000), (7, 23), (1, 3), (64, 1 << 14))

mymax(a, b) = max(a, b)  # not `max`, so integers skip the native `sub_group_reduce_max`


## kernels

# KernelAbstractions' test kernels, under a unique name per round so that `cold` rounds
# compile (Julia, spirv2clc and PoCL) from scratch
function ka_kernels(tag)
    sname, bname = Symbol(:gr_static_, tag), Symbol(:gr_bound_, tag)
    @eval Main begin
        @kernel function $sname(out, @Const(x), op, neutral)
            i = @index(Global, Linear)
            res = @groupreduce(op, x[i], neutral)
            out[i] = res
        end
        @kernel function $bname(out, @Const(x), op, neutral, ::Val{N}) where {N}
            i = @index(Global, Linear)
            res = @groupreduce(op, x[i], neutral; groupsize = N)
            out[i] = res
        end
    end
    return Base.invokelatest(getglobal, Main, sname), Base.invokelatest(getglobal, Main, bname)
end

# KernelInterface's sub-group reduction on its own (the shuffle butterfly), one result per lane
function ki_sgreduce!(out, x, op)
    i = KI.get_global_id().x
    v = @inbounds x[i]
    r = KI.sub_group_reduce(op, v)
    @inbounds out[i] = r
    return
end

# the same algorithm as `@groupreduce` with `KI.sub_group_reduce`, written in OpenCL C,
# which takes Julia's code generation and spirv2clc out of the picture
const CL_SOURCE = """
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
kernel __attribute((intel_reqd_sub_group_size(32)))
void cl_groupreduce(global float *out, global const float *x, uint n) {
  local float storage[256];
  uint gid = get_global_id(0);
  float val = gid < n ? x[gid] : -INFINITY;
  uint width = get_max_sub_group_size();
  uint sgsize = get_sub_group_size();
  uint lane0 = get_sub_group_local_id();
  for (uint mask = 1; mask < width; mask <<= 1) {
    float other = sub_group_shuffle_xor(val, mask);
    if ((lane0 ^ mask) < sgsize) val = fmax(val, other);
  }
  float first = sub_group_shuffle(val, 0);
  val = sgsize == width ? val : first;
  if (lane0 == 0) storage[get_sub_group_id()] = val;
  barrier(CLK_LOCAL_MEM_FENCE);
  float acc = -INFINITY;
  for (uint i = lane0; i < get_num_sub_groups(); i += sgsize) acc = fmax(acc, storage[i]);
  for (uint mask = 1; mask < width; mask <<= 1) {
    float other = sub_group_shuffle_xor(acc, mask);
    if ((lane0 ^ mask) < sgsize) acc = fmax(acc, other);
  }
  first = sub_group_shuffle(acc, 0);
  acc = sgsize == width ? acc : first;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_id() == 0 && lane0 == 0) storage[0] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  float res = storage[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (gid < n) out[gid] = res;
}
"""


## checking

# the reduction of every chunk of `width` consecutive elements, for every element
chunkwise(op, x, width) = [reduce(op, x[(div(i - 1, width) * width + 1):min(div(i - 1, width) * width + width, end)]) for i in eachindex(x)]

# lane ranges, e.g. "0:3,8,10:31"
function ranges(lanes)
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

const stats = Dict{String, Vector{Int}}()  # name => [runs, failures]
const prev_inputs = Dict{String, Vector}()

# compare `got` with the reduction over chunks of `width`, and diagnose the wrong chunks:
# which lanes hold a value that the result doesn't account for (for `max`, the lanes with a
# larger value), whether all lanes of a chunk agree, and whether the result comes from
# elsewhere (another chunk, or the previous input)
function check(name, got, x, width, op; context)
    s = get!(stats, name, [0, 0])
    s[1] += 1
    expected = chunkwise(op, x, width)
    prev = get(prev_inputs, name, nothing)
    prev_inputs[name] = x
    got == expected && return true

    s[2] += 1
    nchunks = cld(length(x), width)
    bad = [c for c in 1:nchunks if got[(c - 1) * width + 1:min(c * width, end)] != expected[(c - 1) * width + 1:min(c * width, end)]]
    report("FAILURE #$(s[2]) of $name ($context) at $(stamp()): $(length(bad))/$nchunks chunks of $width wrong")
    for c in first(bad, 4)
        r = (c - 1) * width + 1:min(c * width, length(x))
        g, xs = got[r], x[r]
        vals = unique(g)
        msg = "  chunk $c (elements $(first(r)):$(last(r))): got $(length(vals) == 1 ? vals[1] : "disagreeing $(first(vals, 8))"), expected $(expected[first(r)])"
        if op === max || op === mymax
            v = minimum(vals)
            msg *= "; lanes above the result: $(ranges(findall(>(v), xs) .- 1))"
            msg *= "; result $(v in xs ? "is" : "is NOT") in the chunk"
            msg *= v in x ? "" : "; NOT in the input"
            prev !== nothing && length(prev) >= last(r) && !(v in xs) && v == maximum(prev[r]) &&
                (msg *= "; equals the previous input's result")
        end
        report(msg)
        report("    got:   $(g[1:min(end, 64)])")
        report("    input: $(xs[1:min(end, 64)])")
    end
    return false
end


## running

function run_ka(kstatic, kbound, b, backend, T, op, neutral; tag)
    for (gs, n) in CONFIGS
        x = T.(rand(1:100, n))
        out = CLArray(fill(zero(T), n))
        Base.invokelatest(Base.invokelatest(kstatic, b, gs), out, CLArray(x), op, neutral; ndrange = n)
        check("ka_static/$backend/$T/$(nameof(op))/$gs", Array(out), x, gs, op; context = tag)

        fill!(out, zero(T))
        Base.invokelatest(Base.invokelatest(kbound, b), out, CLArray(x), op, neutral, Val(256); ndrange = n, workgroupsize = gs)
        check("ka_bound/$backend/$T/$(nameof(op))/$gs", Array(out), x, gs, op; context = tag)
    end
end

function run_ki(b, backend; tag)
    sgw = KI.sub_group_size(b)
    for (gs, n) in CONFIGS
        n = cld(n, gs) * gs  # KI launches aren't masked
        x = Float32.(rand(1:100, n))
        out = CLArray(zeros(Float32, n))
        KI.@launch b ndrange = n workgroupsize = gs ki_sgreduce!(out, CLArray(x), max)
        check("ki_sgreduce/$backend/Float32/$gs", Array(out), x, min(gs, sgw), max; context = tag)
    end
end

const cl_kernel = Ref{Any}(nothing)
function run_cl(; tag)
    if cl_kernel[] === nothing
        prog = cl.build!(cl.Program(; source = CL_SOURCE))
        cl_kernel[] = cl.Kernel(prog, "cl_groupreduce")
    end
    k = cl_kernel[]
    for (gs, n) in CONFIGS
        gs == 1 && continue  # the kernel needs sub-groups of 32
        x = Float32.(rand(1:100, n))
        xb = cl.Buffer(x)
        ob = cl.Buffer(4n)
        cl.set_args!(k, ob, xb, UInt32(n))
        cl.enqueue_kernel(k, cld(n, gs) * gs, gs)
        got = Vector{Float32}(undef, n)
        cl.enqueue_read(pointer(got), ob, 0, 4n; blocking = true)
        cl.finish(cl.queue())
        check("opencl_c/$gs", got, x, gs, max; context = tag)
    end
end

const round_counter = Ref(0)
function round!(b; cold, warm_kernels)
    round_counter[] += 1
    tag = "round $(round_counter[])"
    for backend in (:opencl, :spirv)
        OpenCL.program_backend!(backend) do
            kstatic, kbound = cold ? ka_kernels("$(backend)_$(round_counter[])") : warm_kernels[backend]
            run_ka(kstatic, kbound, b, backend, Float32, max, typemin(Float32); tag)
            # controls: the shuffle path for integers, and the native reduction for `+`
            run_ka(kstatic, kbound, b, backend, Int32, mymax, typemin(Int32); tag)
            run_ka(kstatic, kbound, b, backend, Float32, +, 0.0f0; tag)
            run_ki(b, backend; tag)
        end
    end
    run_cl(; tag)
end

function run_phase(phase, b, warm_kernels)
    report("=== phase $phase on $DRIVER, for $(SECONDS)s, at $(stamp())")
    helpers = Base.Process[]
    stop = Threads.Atomic{Bool}(false)
    gc_task = nothing
    try
        if phase == "cpuload"
            for _ in 1:2 * Sys.CPU_THREADS
                push!(helpers, run(pipeline(`yes`, devnull); wait = false))
            end
        elseif phase == "gc"
            gc_task = Threads.@spawn begin
                garbage = Any[]
                while !stop[]
                    push!(garbage, [rand(1000) for _ in 1:100])
                    length(garbage) > 100 && (empty!(garbage); GC.gc(false))
                    yield()
                end
            end
        elseif phase == "sibling"
            cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__) $DRIVER cold $SECONDS $(joinpath(OUTDIR, "sibling"))`
            push!(helpers, run(pipeline(cmd; stdout = joinpath(OUTDIR, "sibling.log"), stderr = joinpath(OUTDIR, "sibling.log")); wait = false))
        end

        cold = phase in ("cold", "sibling")
        deadline = time() + SECONDS
        rounds = 0
        while time() < deadline
            round!(b; cold, warm_kernels)
            rounds += 1
            phase == "gc" && GC.gc()
        end
        report("=== phase $phase done after $rounds rounds")
    finally
        stop[] = true
        gc_task === nothing || wait(gc_task)
        foreach(p -> kill(p, Base.SIGKILL), helpers)
        foreach(wait, helpers)
    end
end

function main()
    report("=== groupreduce stress test on $DRIVER, phases $(join(PHASES, ",")), $(Threads.nthreads()) threads")
    report("POCL_* environment: $(join(["$k=$v" for (k, v) in ENV if startswith(k, "POCL")], " "))")
    InteractiveUtils.versioninfo()
    OpenCL.versioninfo()
    dev = cl.device()
    report("device: $(dev.name), $(dev.max_compute_units) compute units, max work-group size $(dev.max_work_group_size)")
    b = OpenCLBackend()
    report("sub-group size $(KI.sub_group_size(b)), shuffles of Float32: $(KI.supports_shuffle(b, Float32))")

    warm_kernels = Dict(backend => ka_kernels("warm_$backend") for backend in (:opencl, :spirv))
    for phase in PHASES
        run_phase(phase, b, warm_kernels)
    end

    report("=== summary for $DRIVER ($(join(PHASES, ","))):")
    failures = 0
    for name in sort(collect(keys(stats)))
        runs, fails = stats[name]
        failures += fails
        fails > 0 && report(@sprintf("  %-40s %5d failures in %6d runs", name, fails, runs))
    end
    report("  $(sum(first, values(stats))) checks, $failures failures")
    close(LOG)
    exit(failures > 0 ? 1 : 0)
end

main()
