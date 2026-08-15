#include <math.h>
#include <stdint.h>

extern "C" __global__ void gelu_erf_f32(
    const float *input,
    float *output,
    const uint64_t input_offset,
    const uint64_t element_count,
    const float divisor,
    const float offset,
    const float scale) {
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const float value = input[input_offset + index];
        const float divided = __fdiv_rn(value, divisor);
        const float activated = erff(divided);
        const float shifted = __fadd_rn(activated, offset);
        const float product = __fmul_rn(value, shifted);
        output[index] = __fmul_rn(product, scale);
    }
}
