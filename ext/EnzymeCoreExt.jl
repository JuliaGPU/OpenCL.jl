# compatibility with EnzymeCore
module EnzymeCoreExt

using OpenCL
import OpenCL: GPUCompiler, OpenCLBackend, OpenCLKernels, cl

import EnzymeCore

# the compiler job that kernels for `backend` are compiled with, for Enzyme to compile the
# derivatives of kernels (e.g. by KernelAbstractions' Enzyme extension) and their tapes
function EnzymeCore.compiler_job_from_backend(
        backend::OpenCLBackend, @nospecialize(F::Type), @nospecialize(TT::Type)
    )
    OpenCLKernels.activate(backend)
    mi = GPUCompiler.methodinstance(F, TT)
    return GPUCompiler.CompilerJob(mi, OpenCL.compiler_config(cl.device()))
end

end
