#include <math.h>
#include <stdint.h>
#include "../../cuda_fast_divide.cuh"

__device__ __forceinline__ uint32_t channel_for_index(
    uint32_t index,
    uint32_t spatial_multiplier,
    uint32_t spatial_shift,
    uint32_t channels_multiplier,
    uint32_t channels_shift,
    uint32_t channels) {
    const uint32_t channel_batch =
        fast_divide_u32(index, spatial_multiplier, spatial_shift);
    const uint32_t batch_index =
        fast_divide_u32(channel_batch, channels_multiplier, channels_shift);
    return channel_batch - batch_index * channels;
}

__device__ __forceinline__ float bounded_hard_sigmoid(
    const float value,
    const float alpha,
    const float beta) {
    const float scaled = __fmaf_rn(value, alpha, 0.0f);
    const float shifted = __fmaf_rn(scaled, 1.0f, beta);
    const float lower_bounded = fmaxf(shifted, 0.0f);
    return fminf(lower_bounded, 1.0f);
}

extern "C" __global__ void channel_bias_f32(
    const float *input,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t bias_offset,
    const uint32_t element_count,
    const uint32_t spatial_multiplier,
    const uint32_t spatial_shift,
    const uint32_t channels_multiplier,
    const uint32_t channels_shift,
    const uint32_t channels,
    const float divisor,
    const float offset,
    const float scale) {
    (void)divisor;
    (void)offset;
    (void)scale;
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    const uint32_t channel = channel_for_index(
        index, spatial_multiplier, spatial_shift, channels_multiplier,
        channels_shift, channels);
    output[index] =
        __fadd_rn(input[input_offset + index], bias[bias_offset + channel]);
}

extern "C" __global__ void channel_bias_relu_f32(
    const float *input,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t bias_offset,
    const uint32_t element_count,
    const uint32_t spatial_multiplier,
    const uint32_t spatial_shift,
    const uint32_t channels_multiplier,
    const uint32_t channels_shift,
    const uint32_t channels,
    const float divisor,
    const float offset,
    const float scale) {
    (void)divisor;
    (void)offset;
    (void)scale;
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    const uint32_t channel = channel_for_index(
        index, spatial_multiplier, spatial_shift, channels_multiplier,
        channels_shift, channels);
    const float biased =
        __fadd_rn(input[input_offset + index], bias[bias_offset + channel]);
    output[index] = fmaxf(biased, 0.0f);
}

extern "C" __global__ void channel_bias_gelu_erf_f32(
    const float *input,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t bias_offset,
    const uint32_t element_count,
    const uint32_t spatial_multiplier,
    const uint32_t spatial_shift,
    const uint32_t channels_multiplier,
    const uint32_t channels_shift,
    const uint32_t channels,
    const float divisor,
    const float offset,
    const float scale) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    const uint32_t channel = channel_for_index(
        index, spatial_multiplier, spatial_shift, channels_multiplier,
        channels_shift, channels);
    const float value =
        __fadd_rn(input[input_offset + index], bias[bias_offset + channel]);
    const float divided = __fdiv_rn(value, divisor);
    const float activated = erff(divided);
    const float shifted = __fadd_rn(activated, offset);
    const float product = __fmul_rn(value, shifted);
    output[index] = __fmul_rn(product, scale);
}

extern "C" __global__ void channel_bias_residual_f32(
    const float *input,
    const float *residual,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t residual_offset,
    const uint64_t bias_offset,
    const uint32_t element_count,
    const uint32_t spatial_multiplier,
    const uint32_t spatial_shift,
    const uint32_t channels_multiplier,
    const uint32_t channels_shift,
    const uint32_t channels) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    const uint32_t channel = channel_for_index(
        index, spatial_multiplier, spatial_shift, channels_multiplier,
        channels_shift, channels);
    const float biased =
        __fadd_rn(input[input_offset + index], bias[bias_offset + channel]);
    output[index] = __fadd_rn(biased, residual[residual_offset + index]);
}

extern "C" __global__ void channel_bias_gated_hard_sigmoid_mul_f32(
    const float *multiplicand,
    const float *gate,
    const float *bias,
    float *output,
    const uint64_t multiplicand_offset,
    const uint64_t gate_offset,
    const uint64_t bias_offset,
    const uint32_t element_count,
    const uint32_t spatial_multiplier,
    const uint32_t spatial_shift,
    const uint32_t channels_multiplier,
    const uint32_t channels_shift,
    const uint32_t channels,
    const float alpha,
    const float beta) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    const uint32_t channel = channel_for_index(
        index, spatial_multiplier, spatial_shift, channels_multiplier,
        channels_shift, channels);
    const float biased_gate =
        __fadd_rn(gate[gate_offset + index], bias[bias_offset + channel]);
    const float bounded = bounded_hard_sigmoid(biased_gate, alpha, beta);
    output[index] =
        __fmul_rn(multiplicand[multiplicand_offset + index], bounded);
}

extern "C" __global__ void channel_bias_gated_hard_sigmoid_channel_mul_f32(
    const float *multiplicand,
    const float *gate,
    const float *bias,
    float *output,
    const uint64_t multiplicand_offset,
    const uint64_t gate_offset,
    const uint64_t bias_offset,
    const uint32_t element_count,
    const uint32_t spatial_multiplier,
    const uint32_t spatial_shift,
    const uint32_t channels_multiplier,
    const uint32_t channels_shift,
    const uint32_t channels,
    const float alpha,
    const float beta) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    const uint32_t gate_index =
        fast_divide_u32(index, spatial_multiplier, spatial_shift);
    const uint32_t batch_index =
        fast_divide_u32(gate_index, channels_multiplier, channels_shift);
    const uint32_t channel = gate_index - batch_index * channels;
    const float biased_gate = __fadd_rn(
        gate[gate_offset + gate_index], bias[bias_offset + channel]);
    const float bounded = bounded_hard_sigmoid(biased_gate, alpha, beta);
    output[index] =
        __fmul_rn(multiplicand[multiplicand_offset + index], bounded);
}
