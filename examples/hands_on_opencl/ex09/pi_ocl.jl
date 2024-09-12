#
# Pi reduction
#
# Numeric integration to estimate pi
# Asks the user to select a device at runtime
#
# History: C version written by Tim Mattson, May 2010
#          Ported to the C++ Wrapper API by Benedict R. Gaster, September 2011
#          C++ version Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Ported to Python by Tom Deakin, July 2013
#          Ported to Julia by Jake Bolewski, Nov 2013

using OpenCL

# get the directory of this file
# (used for test runner)
src_dir = dirname(Base.source_path())

#
# Some constant values
const INSTEPS = 512*512*512
const ITERS = 262144

# Set some default values:
# Default number of steps (updated later to device prefereable)
const in_nsteps = INSTEPS

# Default number of iterations
const niters = ITERS

kernelsource = read(joinpath(src_dir, "pi_ocl.cl"), String)
program = cl.Program(source=kernelsource) |> cl.build!

# pi is a julia keyword
pi_kernel = cl.Kernel(program, "pi")

# get the max work group size for the kernel pi on the device
work_group_size = cl.device().max_work_group_size

# now that we know the size of the work_groups, we can set the number
# of work groups, the actual number of steps, and the step size
nwork_groups = in_nsteps ÷ (work_group_size * niters)

if nwork_groups < 1
    # you can get opencl object info through the getproperty syntax
    nwork_groups = cl.device().max_compute_units
    work_group_size = in_nsteps ÷ (nwork_groups * niters)
end

nsteps = work_group_size * niters * nwork_groups
step_size = 1.0 / nsteps

# vector to hold partial sum
h_psum = Vector{Float32}(undef, nwork_groups)

println("$nwork_groups work groups of size $work_group_size.")
println("$nsteps integration steps")

d_partial_sums = CLArray{Float32}(undef, length(h_psum); access=:w)

# start timer
rtime = time()

# Execute the kernel over the entire range of our 1d input data et
# using the maximum number of work group items for this device
# Set the global and local size as tuples
global_size = (nwork_groups * work_group_size,)
local_size  = (work_group_size,)
localmem    = cl.LocalMem(Float32, work_group_size)

clcall(pi_kernel, Tuple{Int32, Float32, cl.LocalMem{Float32}, Ptr{Float32}},
       niters, step_size, localmem, d_partial_sums; global_size, local_size)

cl.copy!(h_psum, d_partial_sums)

# complete the sum and compute final integral value
pi_res = sum(h_psum) * step_size

# stop the timer
rtime = time() - rtime

println("The calculation ran in $rtime secs")
println("pi=$pi_res for $nsteps steps")
