testsdir = dirname(Base.source_path())

for t in [:platform, :context, :device, :cmdqueue, :event, :buffer, :program, :kernel]
    tfile = joinpath(testsdir, "test_$t.jl")
    run(`julia $tfile`)
    # On my crappy laptop, I constantly get "out of host memory" errors
    # when evaling each file, use eval for faster test execution
    #evalfile(tfile)
end

run(`julia $(joinpath(testsdir, "behavior_tests.jl"))`)

cd(joinpath(testsdir, "compiler"))
run(`julia runtests.jl`)

