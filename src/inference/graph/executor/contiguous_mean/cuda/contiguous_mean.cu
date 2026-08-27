#include <stdint.h>

const uint32_t MAXIMUM_THREADS = 1024;

extern "C" __global__ void contiguous_suffix_mean_f32(
    const float *input,
    float *output,
    const uint64_t input_offset,
    const uint32_t reduced_elements,
    const float scale) {
    __shared__ float partial[MAXIMUM_THREADS];
    const uint32_t thread = threadIdx.x;
    const uint32_t row = blockIdx.x;
    const uint32_t start = row * reduced_elements;
    partial[thread] = 0.0f;
    for (uint32_t index = thread; index < reduced_elements; index += blockDim.x) {
        partial[thread] = __fadd_rn(
            partial[thread], input[input_offset + start + index]);
    }

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (thread < stride) {
            partial[thread] =
                __fadd_rn(partial[thread], partial[thread + stride]);
        }
    }
    if (thread == 0) {
        output[row] = __fmaf_rn(partial[0], scale, 0.0f);
    }
}
