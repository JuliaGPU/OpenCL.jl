using FactCheck
using Base.Test

import OpenCL
cl = OpenCL

function always_true(x)
    return true
end

facts("test_fact_check") do
    context("Always true") do
        @fact always_true(10) => true
        @fact always_true(false) => true
        @fact always_true(true) => true
    end
end


facts("test_platform") do
    context("test platform constructor") do
        platforms = cl.platforms()
        @fact platforms != nothing => true
        @fact all([p.id != C_NULL for p in platforms]) => true
        @fact length(platforms) > 0 => true
    end
end

