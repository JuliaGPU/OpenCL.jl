testsdir = dirname(Base.source_path())

info(
"======================================================================
                              Running Compiler Tests
      ======================================================================")

 run(`julia clcompiler.jl`)

 for t in [:clast, :extensions, :structgen, 
           :squares, :gensin, :mersenne, :juliaset, :scholes]
    tfile = joinpath(testsdir, "test_$t.jl")
    run(`julia $tfile`)
end
