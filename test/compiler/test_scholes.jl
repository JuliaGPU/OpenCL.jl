using FactCheck 

import OpenCL
const cl = OpenCL

using OpenCL.Runtime

#TODO: check for double support
device, ctx, queue = cl.create_compute_context()

const RISKFREE = 0.02
const VOLATILITY = 0.30

function cnd(d)
    #TODO: global variables
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    K = 1.0 / (1.0 + 0.2316419 * abs(d))
    ret_val = (RSQRT2PI * exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0.0
        ret_val = 1.0 - ret_val
    end
    return ret_val
end

@clkernel black_scholes(call_result::Vector{Cdouble},
                        put_result::Vector{Cdouble},
                        S::Vector{Cdouble},
                        X::Vector{Cdouble},
                        T::Vector{Cdouble},
                        R::Cdouble,
                        V::Cdouble,
                        len::Int) = begin
    i = get_global_id(0)
    if i >= len
        return
    end
    sqrtT = sqrt(T[i])
    d1 = (log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = cnd(d1)
    cndd2 = cnd(d2)
    expRT = exp((-1. * R) * T[i])
    call_result[i] = S[i] * cndd1 - X[i] * expRT * cndd2
    put_result[i]  = X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1)
    return
end

# TODO: if setindex! is called when array variable is misnamed
# check global scope 

black_scholes_julia(call_result::Vector{Cdouble},
                    put_result::Vector{Cdouble},
                    S::Vector{Cdouble},
                    X::Vector{Cdouble},
                    T::Vector{Cdouble},
                    R::Cdouble,
                    V::Cdouble,
                    len::Int) = begin
    for i = 1:len
        sqrtT = sqrt(T[i])
        d1 = (log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
        d2 = d1 - V * sqrtT
        cndd1 = cnd(d1)
        cndd2 = cnd(d2)
        expRT = exp((-1. * R) * T[i])
        call_result[i] = (S[i] * cndd1 - X[i] * expRT * cndd2)
        put_result[i]  = (X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1))
    end
    return
end

function randfloat(rand_var, low, high)
    return (1.0 - rand_var) * low + rand_var * high
end

facts("Test example black scholes") do
    OPT_N = 100_000
    iterations = 1
    
    stockPrice   = randfloat(rand(Cdouble, OPT_N), 5.0, 30.0)
    optionStrike = randfloat(rand(Cdouble, OPT_N), 1.0, 100.0)
    optionYears  = randfloat(rand(Cdouble, OPT_N), 0.25, 10.0)

    callResultJulia = zeros(Cdouble, OPT_N)
    putResultJulia  = -ones(Cdouble, OPT_N)

    tic()
    for i in 1:iterations
        black_scholes_julia(callResultJulia, putResultJulia, stockPrice,
                            optionStrike, optionYears, RISKFREE, VOLATILITY, OPT_N)
    end
    t = toc()
    info("Julia Time: $((1000 * t) / iterations) msec per iteration")
    
    callResultOpenCL = zeros(Cdouble, OPT_N)
    putResultOpenCL  = -ones(Cdouble, OPT_N)
    
    d_callResult   = cl.Buffer(Cdouble, ctx, :rw, length(callResultOpenCL))
    d_putResult    = cl.Buffer(Cdouble, ctx, :rw, length(putResultOpenCL))
    d_stockPrice   = cl.Buffer(Cdouble, ctx, (:r, :copy), hostbuf=stockPrice)
    d_optionStrike = cl.Buffer(Cdouble, ctx, (:r, :copy), hostbuf=optionStrike)
    d_optionYears  = cl.Buffer(Cdouble, ctx, (:r, :copy), hostbuf=optionYears)
   
    # create a kernel function with queue, global size OPT_N
    black_sholes_ocl = black_sholes[queue, (OPT_N,)]

    tic()
    for i = 1:iterations
        black_sholes_ocl(d_callResult, d_putResult, d_stockPrice, 
                         d_optionStrike, d_optionYears, RISKFREE, VOLATILITY, OPT_N)
        cl.enqueue_barrier(queue)
    end
    cl.copy!(queue, callResultOpenCL, d_callResult)
    cl.copy!(queue, putResultOpenCL,  d_putResult)
    t = toc()
    
    info("OpenCL Time: $((1000 * t) / iterations) msec per iteration")
    
    delta  = abs(callResultJulia - callResultOpenCL)
    L1norm = sum(delta) / sum(abs(callResultOpenCL))
    Maxabs = maximum(delta)

    info("L1 norm (OpenCL): $L1norm")
    info("Max absolute error: $Maxabs")

    @fact L1norm < 1e16 => true
    @fact Maxabs < 1e16 => true
end
