export @memoize, LazyInitialized

"""
    LazyInitialized{T}()

A thread-safe, lazily-initialized wrapper for a value of type `T`. Initialize and fetch the
value by calling `get!`. The constructor is ensured to only be called once.

This type is intended for lazy initialization of e.g. global structures, without using
`__init__`. It is similar to protecting accesses using a lock, but is much cheaper.

"""
struct LazyInitialized{T, F}
    # 0: uninitialized
    # 1: initializing
    # 2: initialized
    guard::Threads.Atomic{Int}
    value::Base.RefValue{T}
    # XXX: use Base.ThreadSynchronizer instead?

    validator::F
end

LazyInitialized{T}(validator = nothing) where {T} =
    LazyInitialized{T, typeof(validator)}(Threads.Atomic{Int}(0), Ref{T}(), validator)

@inline function Base.get!(constructor::Base.Callable, x::LazyInitialized)
    while x.guard[] != 2
        initialize!(x, constructor)
    end
    assume(isassigned(x.value)) # to get rid of the check
    val = x.value[]

    # check if the value is still valid
    if x.validator !== nothing && !x.validator(val)
        Threads.atomic_cas!(x.guard, 2, 0)
        while x.guard[] != 2
            initialize!(x, constructor)
        end
        assume(isassigned(x.value))
        val = x.value[]
    end

    return val
end

@noinline function initialize!(x::LazyInitialized{T}, constructor::F) where {T, F}
    status = Threads.atomic_cas!(x.guard, 0, 1)
    if status == 0
        try
            x.value[] = constructor()::T
            x.guard[] = 2
        catch
            x.guard[] = 0
            rethrow()
        end
    else
        ccall(:jl_cpu_suspend, Cvoid, ())
        # Temporary solution before we have gc transition support in codegen.
        ccall(:jl_gc_safepoint, Cvoid, ())
    end
    return
end


"""
    @memoize [key::T] [maxlen=...] begin
        # expensive computation
    end::T

Low-level, no-frills memoization macro that stores values in a thread-local, typed cache.
The types of the caches are derived from the syntactical type assertions.

The cache consists of two levels, the outer one indexed with the thread index. If no `key`
is specified, the second level of the cache is dropped.

If the the `maxlen` option is specified, the `key` is assumed to be an  integer, and the
secondary cache will be a vector with length `maxlen`. Otherwise, a dictionary is used.
"""
macro memoize(ex...)
    code = ex[end]
    args = ex[1:(end - 1)]

    # decode the code body
    @assert Meta.isexpr(code, :(::))
    rettyp = code.args[2]
    code = code.args[1]

    # decode the arguments
    key = nothing
    if length(args) >= 1
        arg = args[1]
        @assert Meta.isexpr(arg, :(::))
        key = (val = arg.args[1], typ = arg.args[2])
    end
    options = Dict()
    for arg in args[2:end]
        @assert Meta.isexpr(arg, :(=))
        options[arg.args[1]] = arg.args[2]
    end

    # the global cache is an array with one entry per thread. if we don't have to key on
    # anything, that entry will be the memoized new_value, or else a dictionary of values.
    @gensym global_cache

    # in the presence of thread adoption, we need to use the maximum thread ID
    nthreads = :(Threads.maxthreadid())

    # generate code to access memoized values
    # (assuming the global_cache can be indexed with the thread ID)
    if key === nothing
        # if we don't have to key on anything, use the global cache directly
        global_cache_eltyp = :(Union{Nothing, $rettyp})
        ex = quote
            cache = get!($(esc(global_cache))) do
                $global_cache_eltyp[nothing for _ in 1:$nthreads]
            end
            cached_value = @inbounds cache[Threads.threadid()]
            if cached_value !== nothing
                cached_value
            else
                new_value = $(esc(code))::$rettyp
                @inbounds cache[Threads.threadid()] = new_value
                new_value
            end
        end
    elseif haskey(options, :maxlen)
        # if we know the length of the cache, use a fixed-size array
        global_cache_eltyp = :(Vector{Union{Nothing, $rettyp}})
        global_init = :(Union{Nothing, $rettyp}[nothing for _ in 1:$(esc(options[:maxlen]))])
        ex = quote
            cache = get!($(esc(global_cache))) do
                $global_cache_eltyp[$global_init for _ in 1:$nthreads]
            end
            local_cache = @inbounds begin
                tid = Threads.threadid()
                assume(isassigned(cache, tid))
                cache[tid]
            end
            cached_value = @inbounds local_cache[$(esc(key.val))]
            if cached_value !== nothing
                cached_value
            else
                new_value = $(esc(code))::$rettyp
                @inbounds local_cache[$(esc(key.val))] = new_value
                new_value
            end
        end
    else
        # otherwise, fall back to a dictionary
        global_cache_eltyp = :(Dict{$(key.typ), $rettyp})
        global_init = :(Dict{$(key.typ), $rettyp}())
        ex = quote
            cache = get!($(esc(global_cache))) do
                $global_cache_eltyp[$global_init for _ in 1:$nthreads]
            end
            local_cache = @inbounds begin
                tid = Threads.threadid()
                assume(isassigned(cache, tid))
                cache[tid]
            end
            cached_value = get(local_cache, $(esc(key.val)), nothing)
            if cached_value !== nothing
                cached_value
            else
                new_value = $(esc(code))::$rettyp
                local_cache[$(esc(key.val))] = new_value
                new_value
            end
        end
    end

    # define the per-thread cache
    @eval __module__ begin
        const $global_cache = LazyInitialized{Vector{$(global_cache_eltyp)}}() do cache
            length(cache) == $nthreads
        end
    end

    return quote
        $ex
    end
end

