using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

device, ctx, queue = cl.create_compute_context()

@clkernel julia(r::Vector{Float32}, 
                i::Vector{Float32},
                output::Vector{Uint16},
                maxiter::Int32,
                len::Int32) = begin
    gid = get_global_id(0)
    if gid < len 
        nreal = 0.0f0
        real = r[gid]
        imag = i[gid]
        output[gid] = uint16(0)
        for curiter = 1:maxiter
            tmp = abs(real*real + imag*imag)
            if (tmp > 4.0f0) && !(isnan(tmp))
                output[gid] = uint16(curiter - 1)
            end
            nreal = real*real - imag*imag - 0.5f0
            imag = 2.0f0 * real * imag + 0.75f0
            real = nreal
        end
    end
    return
end

test_julia!(r::Vector{Float32}, 
           i::Vector{Float32},
           output::Array{Uint16, 2},
           maxiter::Integer) = begin
    for gid = 1:length(output) 
        nreal = 0.0f0
        real = r[gid]
        imag = i[gid]
        output[gid] = uint16(0)
        for curiter = 1:maxiter
            tmp = abs(real * real + imag * imag)
            if (tmp > 4.0f0) && !(isnan(tmp))
                output[gid] = uint16(curiter - 1)
            end
            nreal = real*real - imag*imag - 0.5f0
            imag = 2.0f0 * real * imag + 0.75f0
            real = nreal
        end
    end
end


@assert isa(julia, cl.Kernel)

function julia_opencl(q::Array{Complex64}, maxiter::Int64)
    r = [real(i) for i in q]
    i = [imag(i) for i in q]
   
    r_buff = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=r)
    i_buff = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf=i)
    o_buff = cl.Buffer(Uint16,  ctx, :rw, length(q))
    
    cl.call(queue, julia, length(q), nothing, 
           r_buff, i_buff, o_buff, int32(maxiter), int32(length(q)))
    out = cl.read(queue, o_buff)
    return reshape(out, size(q))
end

#TODO: diff results using pyplot
facts("Test example generate julia set") do
    w = 512;
    h = 512;
    q = [complex64(r,i) for i=1:-(2.0/w):-1, r=-1.5:(3.0/h):1.5];
    rocl = julia_opencl(q, 200);
    
    rjulia = Array(Uint16, size(q))
    r = Float32[real(i) for i in q]
    i = Float32[imag(i) for i in q]
    test_julia!(r, i, rjulia, 200)
  
    delta  = abs(rjulia - rocl)
    l1norm = sum(delta) / sum(abs(rocl))
    info("L1 norm (OpenCL): $l1norm")
    @fact l1norm < 1.0e-7  => true
end
