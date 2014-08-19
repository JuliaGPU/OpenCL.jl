const tests = [
            "platform"
            "context"
            "device"
            "cmdqueue"
            "event"
            "program"
            "kernel"
            "behaviour"
        ]

const testdir = isdir("test") ? "test" : "."
cd(testdir)

if haskey(ENV, "CODECOVERAGE")
    cmd = `julia --code-coverage`
else
    cmd = `julia`
end

if haskey(ENV, "TRAVIS")
    results = map(tests) do test
        try
            run(`$cmd run.jl $test`)
            return 0
        catch e
            return -1
        end
    end
    all(r -> r == 0, results) ? exit() : exit(-1)
else
    run(`$cmd run.jl $tests`)
end