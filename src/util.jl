"""
Format string using dict-like variables, replacing all accurancies of
`%(key)` with `value`.

Example:
    s = "Hello, %(name)"
    format(s, name="Tom")  ==> "Hello, Tom"
"""
function format(s::String; vars...)
    for (k, v) in vars
        s = replace(s, "%($k)" => v)
    end
    s
end

function build_kernel(program::String, kernel_name::String; vars...)
    src = format(program; vars...)
    p = cl.Program(source=src)
    cl.build!(p)
    return cl.Kernel(p, kernel_name)
end

const cached_kernels = Dict{Any, cl.Kernel}()
const cached_kernels_lock = ReentrantLock()
function get_kernel(program_file::String, kernel_name::String; vars...)
    key = (cl.context(), program_file, kernel_name, Dict(vars))
    return Base.@lock cached_kernels_lock begin
        get!(cached_kernels, key) do
            build_kernel(read(program_file, String), kernel_name; vars...)
        end
    end
end

function versioninfo(io::IO=stdout)
    println(io, "OpenCL.jl version $(pkgversion(@__MODULE__))")
    println(io)

    println(io, "Toolchain:")
    println(io, " - Julia v$(VERSION)")
    for jll in [cl.OpenCL_jll, SPIRV_LLVM_Backend_jll]
        name = string(jll)
        println(io, " - $(name[1:end-4]): $(pkgversion(jll))")
    end
    println(io)

    println(io, "Julia packages:")

    get_module(name::Symbol) = (name, getfield(OpenCL, name))
    function get_module(pkg::Tuple{String, String})
        id = Base.PkgId(Base.UUID(pkg[1]), pkg[2])
        (pkg[2], get(Base.loaded_modules, id, nothing))
    end

    for pkg in [:GPUArrays, :GPUCompiler, ("63c18a36-062a-441e-b654-da1e3ab1ce7c", "KernelAbstractions"),
                 :KernelInterface, :LLVM, :SPIRVIntrinsics, ("627d6b7a-bbe6-5189-83e7-98cc0a5aeadd", "pocl_jll"),
                 ("59abdad9-3cfc-5436-8271-411e8cad6b82", "pocl_next_jll")]
        name, mod = get_module(pkg)
        isnothing(mod) || println(io, "- $(name): $(Base.pkgversion(mod))")
    end
    println(io)

    env = filter(var->startswith(var, "JULIA_OPENCL"), keys(ENV))
    if !isempty(env)
        println(io, "Environment:")
        for var in env
            println(io, "- $var: $(ENV[var])")
        end
        println(io)
    end

    prefs = [
        "default_memory_backend" => load_preference(OpenCL, "default_memory_backend"),
        "llvm_to_spirv_backend" => load_preference(OpenCL, "llvm_to_spirv_backend"),
    ]
    if any(x->!isnothing(x[2]), prefs)
        println(io, "Preferences:")
        for (key, val) in prefs
            if !isnothing(val)
                println(io, "- $key: $val")
            end
        end
        println(io)
    end

    println(io, "Available platforms: ", length(cl.platforms()))
    for platform in cl.platforms()
        println(io, " - $(platform.name)")
        print(io, "   OpenCL $(platform.opencl_version.major).$(platform.opencl_version.minor)")
        if !isempty(platform.version)
            print(io, ", $(platform.version)")
        end
        println(io)

        for device in cl.devices(platform)
            print(io, "   · $(device.name)")

            # show a list of tags
            tags = String[]
            ## memory back-ends
            let
                svm_tags = String[]
                svm_caps = cl.svm_capabilities(device)
                if svm_caps.coarse_grain_buffer
                    push!(svm_tags, "c")
                end
                if svm_caps.fine_grain_buffer
                    push!(svm_tags, "f")
                end
                push!(tags, "svm:"*join(svm_tags, "+"))
            end
            if cl.usm_supported(device)
                usm_tags = String[]
                usm_caps = cl.usm_capabilities(device)
                if usm_caps.host.access
                    push!(usm_tags, "h")
                end
                if usm_caps.device.access
                    push!(usm_tags, "d")
                end
                if usm_caps.shared.access
                    push!(usm_tags, "s")
                end
                push!(tags, "usm:"*join(usm_tags, "+"))
            end
            if cl.bda_supported(device)
                push!(tags, "bda")
            end
            ## relevant extensions
            if in("cl_khr_fp16", device.extensions)
                push!(tags, "fp16")
            end
            if in("cl_khr_fp64", device.extensions)
                push!(tags, "fp64")
            end
            if in("cl_khr_il_program", device.extensions)
                push!(tags, "il")
            end
            if cl.sub_groups_supported(device)
                sg_tags = String[]
                if in("cl_khr_subgroup_shuffle", device.extensions)
                    push!(sg_tags, "shfl")
                end
                # if in("cl_khr_subgroup_ballot", device.extensions)
                #     push!(sg_tags, "blt")
                # end
                push!(tags, "sg:"*join(sg_tags, "+"))
            end
            ## render
            if !isempty(tags)
                print(io, " (", join(tags, ", "), ")")
            end
            println(io)
        end
    end
end
