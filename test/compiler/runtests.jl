testsdir = dirname(Base.source_path())

info(
"======================================================================
                              Running Compiler Tests
      ======================================================================")

 for t in [:clast, :extensions, :structgen, :gensin, :mersenne, :juliaset]
    tfile = joinpath(testsdir, "test_$t.jl")
    run(`julia $tfile`)
end
