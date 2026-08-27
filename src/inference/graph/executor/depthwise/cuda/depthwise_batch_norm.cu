#include <stdint.h>

#include "contiguous.cuh"

__device__ __forceinline__ float apply_batch_norm_values(
    float value,
    float scale,
    float bias,
    float mean,
    float denominator,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  const float centered = __fsub_rn(value, mean);
  const float normalized = __fdiv_rn(centered, denominator);
  const float scaled = __fmul_rn(normalized, scale);
  const float normalized_output = __fadd_rn(scaled, bias);
  if (activation == 0) {
    return normalized_output;
  }
  if (activation == 2) {
    return fmaxf(normalized_output, 0.0f);
  }
  if (activation == 3) {
    const float exponential = expf(-normalized_output);
    const float activation_denominator = __fadd_rn(1.0f, exponential);
    const float gate = __fdiv_rn(1.0f, activation_denominator);
    return __fmul_rn(normalized_output, gate);
  }
  if (activation == 4) {
    const float divided = __fdiv_rn(normalized_output, alpha);
    const float activated = erff(divided);
    const float shifted = __fadd_rn(activated, beta);
    const float product = __fmul_rn(normalized_output, shifted);
    return __fmul_rn(product, gamma);
  }
  const float gate_scaled = __fmaf_rn(normalized_output, alpha, 0.0f);
  const float gate_shifted = __fmaf_rn(gate_scaled, 1.0f, beta);
  const float lower_bounded = fmaxf(gate_shifted, 0.0f);
  const float gate = fminf(lower_bounded, 1.0f);
  return __fmul_rn(normalized_output, gate);
}

__device__ __forceinline__ float apply_batch_norm_contiguous_u32(
    float value,
    uint32_t channel,
    uint32_t channels,
    const float* scale_and_bias,
    const float* mean_and_stddev,
    uint64_t scale_and_bias_offset,
    uint64_t mean_and_stddev_offset,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  const float scale = scale_and_bias[scale_and_bias_offset + channel];
  const float bias =
      scale_and_bias[scale_and_bias_offset + channels + channel];
  const float mean = mean_and_stddev[mean_and_stddev_offset + channel];
  const float stddev =
      mean_and_stddev[mean_and_stddev_offset + channels + channel];
  return apply_batch_norm_values(
      value, scale, bias, mean, stddev, activation, alpha, beta, gamma);
}

extern "C" __global__ void depthwise_conv2d_batch_norm_contiguous_u32_f32(
    const float* input,
    const float* kernel,
    const float* scale_and_bias,
    const float* mean_and_stddev,
    float* output,
    uint32_t batch,
    uint32_t channels,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation,
    uint32_t pad_top,
    uint32_t pad_left,
    uint32_t output_width_multiplier,
    uint32_t output_width_shift,
    uint32_t output_width_divisor,
    uint32_t spatial_multiplier,
    uint32_t spatial_shift,
    uint32_t spatial_divisor,
    uint32_t channels_multiplier,
    uint32_t channels_shift,
    uint32_t channels_divisor,
    uint64_t scale_and_bias_offset,
    uint64_t mean_and_stddev_offset,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t spatial = output_height * output_width;
  const uint32_t elements = batch * channels * spatial;
  if (index >= elements) {
    return;
  }
  uint32_t output_x;
  uint32_t output_y;
  uint32_t channel;
  uint32_t batch_index;
  decode_contiguous_u32_index(
      index, output_width_multiplier, output_width_shift,
      output_width_divisor, spatial_multiplier, spatial_shift,
      spatial_divisor, channels_multiplier, channels_shift, channels_divisor,
      output_x, output_y, channel, batch_index);
  const float convolution = depthwise_accumulate_contiguous_u32(
      input, kernel, batch_index, channel, output_y, output_x, channels,
      input_height, input_width, kernel_height, kernel_width, stride_height,
      stride_width, dilation, pad_top, pad_left);
  output[index] = apply_batch_norm_contiguous_u32(
      convolution, channel, channels, scale_and_bias, mean_and_stddev,
      scale_and_bias_offset, mean_and_stddev_offset, activation,
      alpha, beta, gamma);
}

extern "C" __global__ void
depthwise_conv2d_bias_batch_norm_contiguous_u32_f32(
    const float* input,
    const float* kernel,
    const float* convolution_bias,
    const float* scale_and_bias,
    const float* mean_and_stddev,
    float* output,
    uint32_t batch,
    uint32_t channels,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation,
    uint32_t pad_top,
    uint32_t pad_left,
    uint32_t output_width_multiplier,
    uint32_t output_width_shift,
    uint32_t output_width_divisor,
    uint32_t spatial_multiplier,
    uint32_t spatial_shift,
    uint32_t spatial_divisor,
    uint32_t channels_multiplier,
    uint32_t channels_shift,
    uint32_t channels_divisor,
    uint64_t scale_and_bias_offset,
    uint64_t mean_and_stddev_offset,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t spatial = output_height * output_width;
  const uint32_t elements = batch * channels * spatial;
  if (index >= elements) {
    return;
  }
  uint32_t output_x;
  uint32_t output_y;
  uint32_t channel;
  uint32_t batch_index;
  decode_contiguous_u32_index(
      index, output_width_multiplier, output_width_shift,
      output_width_divisor, spatial_multiplier, spatial_shift,
      spatial_divisor, channels_multiplier, channels_shift, channels_divisor,
      output_x, output_y, channel, batch_index);
  const float convolution = depthwise_accumulate_contiguous_u32(
      input, kernel, batch_index, channel, output_y, output_x, channels,
      input_height, input_width, kernel_height, kernel_width, stride_height,
      stride_width, dilation, pad_top, pad_left);
  const float biased = __fadd_rn(convolution, convolution_bias[channel]);
  output[index] = apply_batch_norm_contiguous_u32(
      biased, channel, channels, scale_and_bias, mean_and_stddev,
      scale_and_bias_offset, mean_and_stddev_offset, activation,
      alpha, beta, gamma);
}
