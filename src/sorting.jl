Base.sort!(x::CLArray; kwargs...) = (AK.sort!(x; kwargs...); return x)
Base.sortperm!(ix::CLArray, x::CLArray; kwargs...) = (AK.sortperm!(ix, x; kwargs...); return ix)
Base.sortperm(x::CLArray; kwargs...) = sortperm!(CLArray(1:length(x)), x; kwargs...)
