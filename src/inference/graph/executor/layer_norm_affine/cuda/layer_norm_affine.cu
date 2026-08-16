#include <math.h>
#include <stdint.h>

extern "C" __global__ void layer_norm_affine_tail_f32(
    const float *centered,
    const float *variance,
    const float *scale,
    const float *bias,
    float *output,
    const uint64_t centered_offset,
    const uint64_t variance_offset,
    const uint64_t scale_offset,
    const uint64_t bias_offset,
    const uint64_t element_count,
    const uint32_t features,
    const float epsilon) {
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const uint32_t flat_index = (uint32_t)index;
        const uint32_t feature = flat_index % features;
        const uint32_t row = flat_index / features;
        const float shifted_variance =
            __fadd_rn(variance[variance_offset + row], epsilon);
        const float denominator = sqrtf(shifted_variance);
        const float normalized =
            __fdiv_rn(centered[centered_offset + index], denominator);
        const float scaled =
            __fmul_rn(normalized, scale[scale_offset + feature]);
        output[index] = __fadd_rn(scaled, bias[bias_offset + feature]);
    }
}
