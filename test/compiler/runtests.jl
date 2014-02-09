testsdir = dirname(Base.source_path())

info(
"======================================================================
                              Running Compiler Tests
      ======================================================================")

 for t in [:clast, :extensions, :structgen]
    tfile = joinpath(testsdir, "test_$t.jl")
    run(`julia $tfile`)
end
