#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float bounded_hard_sigmoid(
    const float value,
    const float alpha,
    const float beta) {
    // Match Candle's two affine kernels and ordered maximum/minimum clamp
    // without allowing reassociation across stages.
    const float scaled = __fmaf_rn(value, alpha, 0.0f);
    const float shifted = __fmaf_rn(scaled, 1.0f, beta);
    const float lower_bounded = fmaxf(shifted, 0.0f);
    return fminf(lower_bounded, 1.0f);
}

extern "C" __global__ void gated_hard_sigmoid_mul_f32(
    const float *multiplicand,
    const float *gate,
    float *output,
    const uint64_t multiplicand_offset,
    const uint64_t gate_offset,
    const uint64_t element_count,
    const uint64_t spatial_elements,
    const float alpha,
    const float beta) {
    (void)spatial_elements;
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const float bounded =
            bounded_hard_sigmoid(gate[gate_offset + index], alpha, beta);
        output[index] =
            __fmul_rn(multiplicand[multiplicand_offset + index], bounded);
    }
}

extern "C" __global__ void gated_hard_sigmoid_channel_mul_f32(
    const float *multiplicand,
    const float *gate,
    float *output,
    const uint64_t multiplicand_offset,
    const uint64_t gate_offset,
    const uint64_t element_count,
    const uint64_t spatial_elements,
    const float alpha,
    const float beta) {
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const uint64_t gate_index = gate_offset + index / spatial_elements;
        const float bounded = bounded_hard_sigmoid(gate[gate_index], alpha, beta);
        output[index] =
            __fmul_rn(multiplicand[multiplicand_offset + index], bounded);
    }
}
