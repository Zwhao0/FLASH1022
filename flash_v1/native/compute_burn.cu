#include <cuda_runtime.h>

// Independent FP32 FMA chains occupy multiple SMs, with a small persistent output.
__global__ void compute_burn(float* output, int rounds) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    float x0 = 0.11f + (index % 31) * 0.001f;
    float x1 = 0.22f, x2 = 0.33f, x3 = 0.44f;
    float x4 = 0.55f, x5 = 0.66f, x6 = 0.77f, x7 = 0.88f;
    #pragma unroll 1
    for (int i = 0; i < rounds; ++i) {
        x0 = fmaf(x0, 0.999999f, 0.000001f);
        x1 = fmaf(x1, 0.999998f, 0.000002f);
        x2 = fmaf(x2, 0.999997f, 0.000003f);
        x3 = fmaf(x3, 0.999996f, 0.000004f);
        x4 = fmaf(x4, 0.999995f, 0.000005f);
        x5 = fmaf(x5, 0.999994f, 0.000006f);
        x6 = fmaf(x6, 0.999993f, 0.000007f);
        x7 = fmaf(x7, 0.999992f, 0.000008f);
    }
    output[index] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

extern "C" int flash_compute_launch(void* stream, void* output, int blocks, int rounds) {
    compute_burn<<<blocks, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<float*>(output), rounds);
    return static_cast<int>(cudaGetLastError());
}
