using FactCheck
using Base.Test

import OpenCL 
const cl = OpenCL

macro throws_pred(ex) FactCheck.throws_pred(ex) end 

facts("OpenCL.Program") do 
    
    test_source = "
    __kernel void sum(__global const float *a,
                      __global const float *b, 
                      __global float *c)
    {
      uint gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    "

    function create_test_program()
        ctx = cl.create_some_context()
        cl.Program(ctx, source=test_source)
    end

    context("OpenCL.Program source constructor") do
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            @fact @throws_pred(cl.Program(ctx, source=test_source)) => (false, "no error")
        end
    end
    
    context("OpenCL.Program info") do
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            
            @fact prg[:context] => ctx
            
            @fact typeof(prg[:devices]) => Vector{cl.Device}
            @fact length(prg[:devices]) > 0 => true 
            @fact device in prg[:devices] => true

            @fact typeof(prg[:source]) => ASCIIString
            @fact prg[:source] => test_source

            #@fact typeof(prg[:binaries]) => Dict{cl.Device, Array{Uint8}}

            @fact prg[:reference_count] > 0 => true
            @fact prg[:build_log] => [device => ""]
         end
    end

    context("OpenCL.Program build") do 
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source)
            @fact @throws_pred(cl.build!(prg)) => (false, "no error")
            # BUILD_SUCCESS undefined in POCL implementation..
            if device[:platform][:name] == "Portable Computing Language"
                warn("Skipping OpenCL.Program build for Portable Computing Language Platform")
                continue
            end
            @fact prg[:build_status][device] => cl.CL_BUILD_SUCCESS 
            # test build by methods chaining
            @fact prg[:build_status][device] => cl.CL_BUILD_SUCCESS 
            @fact prg[:build_log] => [device => "\n"]
        end
    end

    context("OpenCL.Program source code") do
        for device in cl.devices()
           ctx = cl.Context(device)
           prg = cl.Program(ctx, source=test_source)
           @fact prg[:source] => test_source
       end
    end

    context("OpenCL.Program binaries") do
        for device in cl.devices()
            ctx = cl.Context(device)
            prg = cl.Program(ctx, source=test_source) |> cl.build!
            
            @fact device in collect(keys(prg[:binaries])) => true
            binaries = prg[:binaries]
            @fact device in collect(keys(binaries)) => true
            @fact binaries[device] => not(nothing)
            @fact length(binaries[device]) > 0 => true
            prg2 = cl.Program(ctx, binaries=binaries)
            try 
                prg2[:source]
            catch err
                @fact isa(err, cl.CLError) => true
                @fact err.code => -45 
                @fact err.desc => :CL_INVALID_PROGRAM_EXECUTABLE
            end
            @fact prg2[:binaries] == binaries => true
        end
    end
end
