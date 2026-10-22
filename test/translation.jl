@testset "SPIR-V to OpenCL C" begin
    mktempdir() do dir
        # Ordered atomics require OpenCL C 2.0, exercising both the source and error paths.
        assembly = """
            OpCapability Addresses
            OpCapability Kernel
            OpMemoryModel Physical64 OpenCL
            OpEntryPoint Kernel %k "clear"
            %void = OpTypeVoid
            %uint = OpTypeInt 32 0
            %pg = OpTypePointer CrossWorkgroup %uint
            %device = OpConstant %uint 1
            %release = OpConstant %uint 4
            %fnty = OpTypeFunction %void %pg
            %k = OpFunction %void None %fnty
            %flag = OpFunctionParameter %pg
            %entry = OpLabel
            OpAtomicFlagClear %flag %device %release
            OpReturn
            OpFunctionEnd
            """
        input = joinpath(dir, "input.spvasm")
        binary = joinpath(dir, "input.spv")
        write(input, assembly)
        run(`$(OpenCL.SPIRV_Tools_jll.spirv_as()) --target-env spv1.4 $input -o $binary`)
        spirv = read(binary)
        for version in (v"2.0", v"3.0")
            source = OpenCL.translate_spirv(spirv, v"1.4", version)
            @test occursin("atomic_flag_clear_explicit", source)
            cl_std = "CL$(version.major).$(version.minor)"
            # The tool prints the source plus a newline, in text mode (CRLF on Windows).
            expected = readlines(`$(OpenCL.spirv2clc_jll.spirv2clc()) --spirv-version=1.4 --cl-std=$cl_std $binary`)
            @test readlines(IOBuffer(source * "\n")) == expected
        end
        # Also exercise successful translation with the OpenCL C 1.2 target.
        write(input, replace(assembly, "OpAtomicFlagClear %flag %device %release" =>
                                       "OpStore %flag %device"))
        run(`$(OpenCL.SPIRV_Tools_jll.spirv_as()) --target-env spv1.4 $input -o $binary`)
        source = OpenCL.translate_spirv(read(binary), v"1.4", v"1.2")
        expected = readlines(`$(OpenCL.spirv2clc_jll.spirv2clc()) --spirv-version=1.4 --cl-std=CL1.2 $binary`)
        @test readlines(IOBuffer(source * "\n")) == expected

        # Returned Julia strings must survive subsequent translations and garbage collection.
        sources = [OpenCL.translate_spirv(spirv, v"1.4", v"2.0") for _ in 1:10]
        GC.gc()
        @test all(==(first(sources)), sources)

        dumps = joinpath(dir, "dumps")
        withenv("JULIA_OPENCL_DUMP_DIR" => dumps) do
            @test_throws "ordered atomics require OpenCL C 2.0" OpenCL.translate_spirv(spirv, v"1.4", v"1.2")
            @test read(only(readdir(dumps; join=true))) == spirv
            @test_throws "not a multiple of 4 bytes" OpenCL.translate_spirv(UInt8[1, 2, 3], v"1.4", v"1.2")
            @test_throws "Invalid SPIR-V magic number" OpenCL.translate_spirv(zeros(UInt8, 24), v"1.4", v"1.2")
            @test_throws "unsupported SPIR-V version" OpenCL.translate_spirv(spirv, v"2.0", v"2.0")
            @test_throws "unsupported OpenCL C version" OpenCL.translate_spirv(spirv, v"1.4", v"2.1")
        end
    end
end
