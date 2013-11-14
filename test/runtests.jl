testsdir = dirname(Base.source_path())

for t in [:platform, :context, :device, :cmdqueue, :event, :buffer, :program, :kernel]
    tfile = joinpath(testsdir, "test_$t.jl")
    run(`julia $tfile`)
end

run(`julia $(joinpath(testsdir, "behavior_tests.jl"))`)
