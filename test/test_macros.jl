using FactCheck
using Base.Test

import OpenCL
const cl = OpenCL

facts("OpenCL.Macros") do

    context("OpenCL.Macros version platform") do
        for platform in cl.platforms()

            version = cl.opencl_version(platform)

            v11 = cl.@min_v11? platform true : false
            v12 = cl.@min_v12? platform true : false
            v20 = cl.@min_v20? platform true : false

            if version == v"1.0"
                @fact v11 => false
                @fact v12 => false
                @fact v20 => false

            elseif version == v"1.1"
                @fact v11 => true
                @fact v12 => false
                @fact v20 => false

            elseif version == v"1.2"
                @fact v11 => true
                @fact v12 => true
                @fact v20 => false

            elseif version == v"2.0"
                @fact v11 => true
                @fact v12 => true
                @fact v20 => true

            end
        end
    end

    context("OpenCL.Macros version device") do
        for platform in cl.platforms()
            for device in cl.devices(platform)
                version = cl.opencl_version(device)

                v11 = cl.@min_v11? device true : false
                v12 = cl.@min_v12? device true : false
                v20 = cl.@min_v20? device true : false

                if version == v"1.0"
                    @fact v11 => false
                    @fact v12 => false
                    @fact v20 => false

                elseif version == v"1.1"
                    @fact v11 => true
                    @fact v12 => false
                    @fact v20 => false

                elseif version == v"1.2"
                    @fact v11 => true
                    @fact v12 => true
                    @fact v20 => false

                elseif version == v"2.0"
                    @fact v11 => true
                    @fact v12 => true
                    @fact v20 => true

                end
            end
        end
    end
end

