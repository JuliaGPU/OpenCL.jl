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
            clmod = OpenCL.Compiler.CLModule()
            # Range1
            r   = $func(1:10)
            rty = typeof(r)
            @fact rty => Range1{$ty}
            structgen!(clmod, rty)
            src = clsource(clmod.structs[rty])
            @fact src => $("typedef struct {{\n\t$cty start;\n\tlong len;\n}} Range1_$ty;\n")
            
            # Range
            r   = $func(1:1:10)
            rty = typeof(r)
            @fact rty => Range{$ty}
            structgen!(clmod, rty)
            src = clsource(clmod.structs[rty])
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
                clmod = OpenCL.Compiler.CLModule()
                ty = Test1{$xty, $yty}
                structgen!(clmod, ty)
                src = clsource(clmod.structs[ty])
                @fact src => $("typedef struct {{\n\t$xcty x;\n\t$ycty y;\n}} Test1_$(xty)_$(yty);\n")
            end
        end
    end
end

type Test2{X, Y}
    x::X
    y::Ptr{Y}
end

type Test3
    x::Int
    y::Ptr{Test2{Int32, Int32}}
end

facts("Structs with pointers") do
    for xty in [Int64, Uint64, Int32, Uint32, Int16, Uint16, Int8, Uint8]
        for yty in [Int64, Uint64, Int32, Uint32, Int16, Uint16, Int8, Uint8]
            xcty = symbol(ctypename(xty))
            ycty = symbol(ctypename(Ptr{yty}))
            @eval begin
                clmod = OpenCL.Compiler.CLModule()
                ty = Test2{$xty, $yty}
                structgen!(clmod, ty)
                src = clsource(clmod.structs[ty])
                @fact src => $("typedef struct {{\n\t$xcty x;\n\t$ycty y;\n}} Test2_$(xty)_$(yty);\n")
            end
        end
    end
    clmod = OpenCL.Compiler.CLModule()
    structgen!(clmod, Test3)
    @fact clsource(clmod.structs[Test3]) => "typedef struct {{\n\tlong x;\n\tTest2_Int32_Int32 * y;\n}} Test3;\n"
end

type Test4{T}
    x::T
end

type Test5{T}
    x::Test4{T}
end

facts("Structs of structs") do
    clmod = OpenCL.Compiler.CLModule()
    ty = Test5{Int32}
    structgen!(clmod, ty)
    @fact collect(keys(clmod.structs)) => [Test4{Int32}, Test5{Int32}]
    src = clsource(clmod.structs[ty])
    @fact src => "typedef struct {{\n\tTest4_Int32 x;\n}} Test5_Int32;\n"
end

type Test6
    x::Test6
end

facts("Error on self referential field types") do
    clmod = OpenCL.Compiler.CLModule()
    @fact_throws OpenCL.Compiler.isvalid_clstruct(Test5)
end

type NoFields end

type Test7
    x::NoFields
end

facts("Error with struct of structs with no fields") do
    clmod = OpenCL.Compiler.CLModule()
    @fact_throws OpenCL.Compiler.isvalid_clstruct(Test6)
end

type Test8
    x::Any
end

type Test9
    x::Test7
end

type Test10
    x::Ptr{Any}
end

facts("Error with struct that contains invalid OpenCL type for field") do
    clmod = OpenCL.Compiler.CLModule()
    @fact_throws OpenCL.Compiler.isvalid_clstruct(Test7)
    @fact_throws OpenCL.Compiler.isvalid_clstruct(Test8)
    @fact_throws OpenCL.Compiler.isvalid_clstruct(Test9)
end
