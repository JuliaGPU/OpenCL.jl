using OpenCL
using Documenter

DocMeta.setdocmeta!(OpenCL, :DocTestSetup, :(using OpenCL, pocl_jll); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [OpenCL],
    authors = "Jake Bolewski, JuliaHub and other contributors",
    repo = "https://github.com/JuliaGPU/OpenCL.jl/blob/{commit}{path}#{line}",
    sitename = "OpenCL.jl",
    format = Documenter.HTML(; canonical = "https://JuliaGPU.github.io/OpenCL.jl"),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/JuliaGPU/OpenCL.jl")
