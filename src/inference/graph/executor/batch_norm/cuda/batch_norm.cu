#include <stdint.h>
#include "../../cuda_fast_divide.cuh"

extern "C" __global__ void prepare_batch_norm_mean_stddev_f32(
    const float* mean_and_variance,
    float* mean_and_stddev,
    uint64_t input_offset,
    uint32_t channels,
    float epsilon) {
  const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  mean_and_stddev[channel] = mean_and_variance[input_offset + channel];
  const float variance =
      mean_and_variance[input_offset + channels + channel];
  const float adjusted_variance = __fadd_rn(variance, epsilon);
  mean_and_stddev[channels + channel] = sqrtf(adjusted_variance);
}

extern "C" __global__ void batch_norm_f32(
    const float* input,
    const float* scale_and_bias,
    const float* mean_and_stddev,
    float* output,
    uint64_t input_offset,
    uint64_t scale_and_bias_offset,
    uint64_t mean_and_stddev_offset,
    uint32_t elements,
    uint32_t spatial_multiplier,
    uint32_t spatial_shift,
    uint32_t channels_multiplier,
    uint32_t channels_shift,
    uint32_t channels,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const uint32_t channel_batch =
      fast_divide_u32(index, spatial_multiplier, spatial_shift);
  const uint32_t batch_index =
      fast_divide_u32(channel_batch, channels_multiplier, channels_shift);
  const uint32_t channel = channel_batch - batch_index * channels;
  const float scale = scale_and_bias[scale_and_bias_offset + channel];
  const float bias =
      scale_and_bias[scale_and_bias_offset + channels + channel];
  const float mean = mean_and_stddev[mean_and_stddev_offset + channel];
  const float stddev =
      mean_and_stddev[mean_and_stddev_offset + channels + channel];
  const float centered = __fsub_rn(input[input_offset + index], mean);
  const float normalized = __fdiv_rn(centered, stddev);
  const float scaled = __fmul_rn(normalized, scale);
  const float normalized_output = __fadd_rn(scaled, bias);
  if (activation == 0) {
    output[index] = normalized_output;
    return;
  }
  if (activation == 2) {
    output[index] = fmaxf(normalized_output, 0.0f);
    return;
  }
  if (activation == 5) {
    const float exponential = expf(-normalized_output);
    const float denominator = __fadd_rn(1.0f, exponential);
    output[index] = __fdiv_rn(1.0f, denominator);
    return;
  }
  if (activation == 3) {
    const float exponential = expf(-normalized_output);
    const float denominator = __fadd_rn(1.0f, exponential);
    const float gate = __fdiv_rn(1.0f, denominator);
    output[index] = __fmul_rn(normalized_output, gate);
    return;
  }
  if (activation == 4) {
    const float divided = __fdiv_rn(normalized_output, alpha);
    const float activated = erff(divided);
    const float shifted = __fadd_rn(activated, beta);
    const float product = __fmul_rn(normalized_output, shifted);
    output[index] = __fmul_rn(product, gamma);
    return;
  }
  const float gate_scaled = __fmaf_rn(normalized_output, alpha, 0.0f);
  const float gate_shifted = __fmaf_rn(gate_scaled, 1.0f, beta);
  const float lower_bounded = fmaxf(gate_shifted, 0.0f);
  const float gate = fminf(lower_bounded, 1.0f);
  output[index] = __fmul_rn(normalized_output, gate);
}

extern "C" __global__ void batch_norm_in_place_f32(
    float* values,
    const float* scale_and_bias,
    const float* mean_and_stddev,
    uint64_t scale_and_bias_offset,
    uint64_t mean_and_stddev_offset,
    uint32_t elements,
    uint32_t spatial_multiplier,
    uint32_t spatial_shift,
    uint32_t channels_multiplier,
    uint32_t channels_shift,
    uint32_t channels,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const uint32_t channel_batch =
      fast_divide_u32(index, spatial_multiplier, spatial_shift);
  const uint32_t batch_index =
      fast_divide_u32(channel_batch, channels_multiplier, channels_shift);
  const uint32_t channel = channel_batch - batch_index * channels;
  const float scale = scale_and_bias[scale_and_bias_offset + channel];
  const float bias =
      scale_and_bias[scale_and_bias_offset + channels + channel];
  const float mean = mean_and_stddev[mean_and_stddev_offset + channel];
  const float stddev =
      mean_and_stddev[mean_and_stddev_offset + channels + channel];
  const float centered = __fsub_rn(values[index], mean);
  const float normalized = __fdiv_rn(centered, stddev);
  const float scaled = __fmul_rn(normalized, scale);
  const float normalized_output = __fadd_rn(scaled, bias);
  if (activation == 0) {
    values[index] = normalized_output;
    return;
  }
  if (activation == 2) {
    values[index] = fmaxf(normalized_output, 0.0f);
    return;
  }
  if (activation == 5) {
    const float exponential = expf(-normalized_output);
    const float denominator = __fadd_rn(1.0f, exponential);
    values[index] = __fdiv_rn(1.0f, denominator);
    return;
  }
  if (activation == 3) {
    const float exponential = expf(-normalized_output);
    const float denominator = __fadd_rn(1.0f, exponential);
    const float gate = __fdiv_rn(1.0f, denominator);
    values[index] = __fmul_rn(normalized_output, gate);
    return;
  }
  if (activation == 4) {
    const float divided = __fdiv_rn(normalized_output, alpha);
    const float activated = erff(divided);
    const float shifted = __fadd_rn(activated, beta);
    const float product = __fmul_rn(normalized_output, shifted);
    values[index] = __fmul_rn(product, gamma);
    return;
  }
  const float gate_scaled = __fmaf_rn(normalized_output, alpha, 0.0f);
  const float gate_shifted = __fmaf_rn(gate_scaled, 1.0f, beta);
  const float lower_bounded = fmaxf(gate_shifted, 0.0f);
  const float gate = fminf(lower_bounded, 1.0f);
  values[index] = __fmul_rn(normalized_output, gate);
}
