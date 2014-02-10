module Runtime

import ..OpenCL
import ..Compiler
import ..SourceGen

export @clkernel, get_global_id, get_global_size


function get_global_id(x)
    x + 2
    y + 3
    return uint32(x)::Uint32
end

function get_global_size(x)
    x + 2
    y + 3
    return uint32(x)::Uint32
end

uncompressed_ast(l::LambdaStaticData) = begin
    isa(l.ast,Expr) ? l.ast : ccall(:jl_uncompress_ast, Any, (Any,Any), l, l.ast) 
end

#function _generate_kernel(func)
macro clkernel(func)
    f, n, prg = gensym("func"), gensym("n"), gensym("prg")
    
    orig_name = func.args[1].args[1]
    func.args[1].args[1] = symbol(f)
    
    quote
        local func = eval($(esc(func)))
        # lookup method name from method table
        local name = func.env.name
        if length(func.env) != 1
            error("more than one kernel with name $name")
        end
        # lookup method signature from first method in method table
        local typs = func.env.defs.sig
        for ty in typs
            if !isleaftype(ty)
                error("function signature nonleaftype $ty")
            end
        end
        local exprs = code_typed(func, typs)
        if length(exprs) == 0
            error("function could not be compiled for attribute types:: $typs")
        end
        if length(exprs) > 1
            error("more than one typed ast produced!")
        end
        local expr = first(exprs)
        
        kern_ctx, kernel = Compiler.build_kernel($("$orig_name"), expr)
        local io  = IOBuffer()
        print(io, "#pragma OPENCL EXTENSION cl_amd_printf : enable\n")
        print(io, "typedef struct Range {long start; long step; long len;} Range;\n")
        for n in unique(keys(kern_ctx.funcs))
            SourceGen.clsource(io, kern_ctx.funcs[n])
            println(io)
        end
        SourceGen.clsource(io, kernel)
        local src = bytestring(io.data)
        # TODO: return a fucntion that takes a context
        # build the source and store in global cache
        local prg  = OpenCL.Program($(esc(:ctx)), source=src) |> OpenCL.build!
        const $(esc(orig_name)) = OpenCL.Kernel(prg, $("$orig_name"))
    end
end

#macro clkernel(func)
#    _generate_kernel(func)
#end

end
