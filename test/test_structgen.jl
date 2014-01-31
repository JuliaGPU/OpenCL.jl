using FactCheck
using Base.Test

import OpenCL
const cl = OpenCL

using OpenCL.CLAst

using OpenCL.Compiler
using OpenCL.SourceGen

ctypename(t) = begin
    sprint() do io
        clprint(io, t, 0)
    end
end

facts("Range generation") do
    for ty in [Int64, Uint64, Int32, Uint32, Int16, Uint16, Int8, Uint8]
        func = symbol(lowercase(string(ty)))
        cty = symbol(ctypename(ty))
        
        @eval begin
            # Range1
            r = $func(1:10)
            @fact typeof(r) => Range1{$ty}
            src = clsource(structgen(typeof(r)))
            @fact src => $("typedef struct {{\n\t$cty start;\n\tlong len;\n}} Range1_$ty;\n")
            
            # Range
            r = $func(1:1:10)
            @fact typeof(r) => Range{$ty}
            src = clsource(structgen(typeof(r)))
            @fact src => $("typedef struct {{\n\t$cty start;\n\t$cty step;\n\tlong len;\n}} Range_$ty;\n")
        end
    end
end

type Test1{X, Y}
    x::X
    y::Y
end

facts("Structs with parametric types") do
    for xty in [Int64, Uint64, Int32, Uint32, Int16, Uint16, Int8, Uint8]
        for yty in [Int64, Uint64, Int32, Uint32, Int16, Uint16, Int8, Uint8]
            xcty = symbol(ctypename(xty))
            ycty = symbol(ctypename(yty))
            @eval begin
                src = clsource(structgen(Test1{$xty, $yty}))
                @fact src => $("typedef struct {{\n\t$xcty x;\n\t$ycty y;\n}} Test1_$(xty)_$(yty);\n")
            end
        end
    end
end

type Test2{X, Y}
    x::X
    y::Ptr{Y}
end

facts("Structs with pointers") do
    for xty in [Int64, Uint64, Int32, Uint32, Int16, Uint16, Int8, Uint8]
        for yty in [Int64, Uint64, Int32, Uint32, Int16, Uint16, Int8, Uint8]
            xcty = symbol(ctypename(xty))
            ycty = symbol(ctypename(Ptr{yty}))
            @eval begin
                src = clsource(structgen(Test2{$xty, $yty}))
                @fact src => $("typedef struct {{\n\t$xcty x;\n\t$ycty y;\n}} Test2_$(xty)_$(yty);\n")
            end
        end
    end
end

type Test3{T}
    x::T
end

type Test4{T}
    x::Test3{T}
end

facts("Structs of structs") do
    src = clsource(structgen(Test4{Int32}))
    @fact src => "typedef struct {{\n\tTest3_Int32 x;\n}} Test4_Int32;\n"
end

type Test5
    x::Test5
end

facts("Error on self referential field types") do
    @fact_throws structgen(Test5)
end
