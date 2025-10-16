using SPIRV_LLVM_Translator_jll
using IOCapture

@testset "execution" begin

@testset "@opencl" begin

dummy() = nothing

@test_throws UndefVarError @opencl undefined()
@test_throws MethodError @opencl dummy(1)


@testset "launch configuration" begin
    @opencl dummy()

    global_size = 1
    @opencl global_size dummy()
    @opencl global_size=1 dummy()
    @opencl global_size=(1,1) dummy()
    @opencl global_size=(1,1,1) dummy()

    local_size = 1
    @opencl global_size local_size dummy()
    @opencl global_size=1 local_size=1 dummy()
    @opencl global_size=(1,1) local_size=(1,1) dummy()
    @opencl global_size=(1,1,1) local_size=(1,1,1) dummy()

    @test_throws ArgumentError @opencl global_size=(1,) local_size=(1,1) dummy()
    @test_throws InexactError @opencl global_size=(-2) dummy()
    @test_throws InexactError @opencl local_size=(-2) dummy()
end

@testset "launch=false" begin
    # XXX: how are svm_pointers handled here?
    k = @opencl launch=false dummy()
    k()
    k(; global_size=1)
end

@testset "inference" begin
    foo() = @opencl dummy()
    @inferred foo()

    # with arguments, we call OpenCL.kernel_convert
    kernel(a) = return
    bar(a) = @opencl kernel(a)
    @inferred bar(CLArray([1]))
end


@testset "reflection" begin
    OpenCL.code_lowered(dummy, Tuple{})
    OpenCL.code_typed(dummy, Tuple{})
    OpenCL.code_warntype(devnull, dummy, Tuple{})
    OpenCL.code_llvm(devnull, dummy, Tuple{})
    OpenCL.code_native(devnull, dummy, Tuple{})

    @device_code_lowered @opencl dummy()
    @device_code_typed @opencl dummy()
    @device_code_warntype io=devnull @opencl dummy()
    @device_code_llvm io=devnull @opencl dummy()
    @device_code_native io=devnull @opencl dummy()

    mktempdir() do dir
        @device_code dir=dir @opencl dummy()
    end

    @test_throws ErrorException @device_code_lowered nothing

    # make sure kernel name aliases are preserved in the generated code
    @test occursin("dummy", sprint(io->(@device_code_llvm io=io optimize=false @opencl dummy())))
    @test occursin("dummy", sprint(io->(@device_code_llvm io=io @opencl dummy())))
    @test occursin("dummy", sprint(io->(@device_code_native io=io @opencl dummy())))

    # make sure invalid kernels can be partially reflected upon
    let
        invalid_kernel() = throw()
        @test_throws OpenCL.InvalidIRError @opencl invalid_kernel()
        @test_throws OpenCL.InvalidIRError IOCapture.capture() do
            @device_code_warntype @opencl invalid_kernel()
        end
        c = IOCapture.capture() do
            try
                @device_code_warntype @opencl invalid_kernel()
            catch
            end
        end
        @test occursin("Body::Union{}", c.output)
    end

    # set name of kernel
    @test occursin("mykernel", sprint(io->(@device_code_llvm io=io begin
        @opencl name="mykernel" dummy()
    end)))

    @test OpenCL.return_type(identity, Tuple{Int}) === Int
    @test OpenCL.return_type(sin, Tuple{Float32}) === Float32
    @test OpenCL.return_type(getindex, Tuple{CLDeviceArray{Float32,1,AS.CrossWorkgroup},Int32}) === Float32
    @test OpenCL.return_type(getindex, Tuple{Base.RefValue{Integer}}) === Integer
end

end

###############################################################################

@testset "argument passing" begin

function memset(a, val)
    gid = get_global_id(1)
    @inbounds a[gid] = val
    return
end

a = CLArray{Int}(undef, 10)
@opencl global_size=length(a) memset(a, 42)
@test all(Array(a) .== 42)

end

@testset "broadcasting" begin
    a = rand(Float32, 2, 3)
    b = rand(Float32, 2)

    c = a .+ b
    a_cl, b_cl = CLArray(a), CLArray(b)
    c_cl = a_cl .+ b_cl
    @test Array(c_cl) == c
    @test c_cl isa CLArray{Float32, 2, OpenCL.memory_type()}

    if cl.usm_supported(cl.device())
        a_cl, b_cl = CLMatrix{Float32, cl.UnifiedSharedMemory}(a), CLVector{Float32, OpenCL.memory_type()}(b)
        c_cl = a_cl .+ b_cl
        @test Array(c_cl) == c
        @test c_cl isa CLArray{Float32, 2, cl.UnifiedSharedMemory}
    end
end

@testset "backends" begin
    llvm_backend_llvm = sprint() do io
        OpenCL.code_llvm(io, () -> nothing, (); dump_module = true, backend = :llvm)
    end
    if Int === Int64
        @test occursin("target triple = \"spirv64-unknown-unknown-unknown\"", llvm_backend_llvm)
    end

    llvm_backend_khronos = sprint() do io
        OpenCL.code_llvm(io, () -> nothing, (); dump_module = true, backend = :khronos)
    end
    if Int === Int64
        @test occursin("target triple = \"spir64-unknown-unknown\"", llvm_backend_khronos)
    end
end

end
