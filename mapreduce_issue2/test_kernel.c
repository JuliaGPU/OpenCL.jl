#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Struct definitions matching the LLVM IR
typedef struct {
    int64_t stop;
} OneTo;

typedef struct {
    void* ptr;           // i8 addrspace(1)*
    int64_t maxsize;     // i64
    int64_t dims[1];     // [1 x i64]
    int64_t len;         // i64
} CLDeviceArray;

typedef struct {
    int64_t start[1];    // [1 x i64]
    int64_t stop[1];     // [1 x i64]
} IndexRange;

typedef struct {
    CLDeviceArray array;
    IndexRange ranges[2];
} BroadcastedArray;

// Helper function to check OpenCL errors
#define CHECK_CL(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error at %s: %d\n", msg, err); \
        exit(1); \
    }

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <spirv_file>\n", argv[0]);
        return 1;
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_CL(err, "clGetPlatformIDs");

    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        // Try CPU if GPU not available
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        CHECK_CL(err, "clGetDeviceIDs");
    }

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    // Read SPIR-V file
    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* spirv = malloc(size);
    fread(spirv, 1, size, f);
    fclose(f);

    // Create program from SPIR-V
    program = clCreateProgramWithIL(context, spirv, size, &err);
    CHECK_CL(err, "clCreateProgramWithIL");
    free(spirv);

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        CHECK_CL(err, "clBuildProgram");
    }

    // Create kernel
    kernel = clCreateKernel(program, "_Z6kernel5OneToI5Int64E11BroadcastedI12CLArrayStyleILi1E19UnifiedDeviceMemoryEv5tuple5TupleI13CLDeviceArrayI7Float32Li1ELi1EE9EachIndexIS0_Li1ES1_EEE", &err);
    CHECK_CL(err, "clCreateKernel");

    // Prepare test data
    const int N = 2;
    float data[N];
    for (int i = 0; i < N; i++) {
        data[i] = (float)(i + 1);
    }

    // Create buffer for array data
    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(data), data, &err);
    CHECK_CL(err, "clCreateBuffer");

    // Set up OneTo struct (argument 0)
    OneTo oneto;
    oneto.stop = N;

    // Set up BroadcastedArray struct (argument 1)
    BroadcastedArray broadcast;
    broadcast.array.ptr = NULL;  // Will be set by OpenCL as the buffer
    broadcast.array.maxsize = N;  // 1-based indexing
    broadcast.array.dims[0] = N;
    broadcast.array.len = N;
    broadcast.ranges[0].start[0] = 1;
    broadcast.ranges[0].stop[0] = N;
    broadcast.ranges[1].start[0] = 1;
    broadcast.ranges[1].stop[0] = N;

    // We need to pass the buffer pointer through a host-side struct
    // Create a buffer for the broadcast struct
    cl_mem broadcast_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          sizeof(BroadcastedArray), NULL, &err);
    CHECK_CL(err, "clCreateBuffer for broadcast");

    // Update the ptr field to point to the actual data buffer
    // Note: We need to handle this differently for device pointers
    cl_mem ptr_as_mem = buf;
    memcpy(&broadcast.array.ptr, &ptr_as_mem, sizeof(cl_mem));

    err = clEnqueueWriteBuffer(queue, broadcast_buf, CL_TRUE, 0,
                               sizeof(BroadcastedArray), &broadcast, 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueWriteBuffer");

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(OneTo), &oneto);
    CHECK_CL(err, "clSetKernelArg 0");

    err = clSetKernelArg(kernel, 1, sizeof(BroadcastedArray), &broadcast);
    CHECK_CL(err, "clSetKernelArg 1");

    // Execute kernel
    size_t global_size = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueNDRangeKernel");

    // Wait for completion
    err = clFinish(queue);
    CHECK_CL(err, "clFinish");

    // Read results
    err = clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(data), data, 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueReadBuffer");

    printf("Results:\n");
    for (int i = 0; i < N; i++) {
        printf("data[%d] = %f\n", i, data[i]);
    }

    // Cleanup
    clReleaseMemObject(buf);
    clReleaseMemObject(broadcast_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
