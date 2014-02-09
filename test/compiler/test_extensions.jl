using FactCheck
using Base.Test

import OpenCL
const cl = OpenCL

using OpenCL.CLAst

using OpenCL.Compiler
using OpenCL.SourceGen

function test1()
end

function test1()
    a = 1.0
end

function test2()
    a = 1.0f0
end

function test3()
    a = 1.0f0
    test1()
end

facts("Test double support platform extension") do
    # check module for correct extensions
    @fact true => true
    @fact true => true
    @fact true => true
    # check source gen for correct extensions
    @fact true => true
end

function test4()
    a = 1.0
    clprintf("%d\n", a)
end

function test5()
    a = 1.0f0
    clprintf("%d\n", a)
end

facts("Test printf platform extension") do
    @fact true => true
    @fact true => true
    # check source gen for correct extensions
    @fact => true
end
