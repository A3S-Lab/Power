#include <stdint.h>

extern "C" __global__ void im2col_contiguous_u32_f32(
    const float* input,
    float* output,
    uint32_t output_elements,
    uint64_t input_offset,
    uint32_t input_channels,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride,
    uint32_t dilation,
    uint32_t pad_top,
    uint32_t pad_left) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= output_elements) {
    return;
  }

  uint32_t quotient = index;
  const uint32_t kernel_x = quotient % kernel_width;
  quotient /= kernel_width;
  const uint32_t kernel_y = quotient % kernel_height;
  quotient /= kernel_height;
  const uint32_t channel = quotient % input_channels;
  quotient /= input_channels;
  const uint32_t output_x = quotient % output_width;
  quotient /= output_width;
  const uint32_t output_y = quotient % output_height;
  const uint32_t batch = quotient / output_height;

  const uint32_t padded_y = output_y * stride + kernel_y * dilation;
  const uint32_t padded_x = output_x * stride + kernel_x * dilation;
  if (padded_y < pad_top || padded_x < pad_left) {
    output[index] = 0.0f;
    return;
  }
  const uint32_t input_y = padded_y - pad_top;
  const uint32_t input_x = padded_x - pad_left;
  if (input_y >= input_height || input_x >= input_width) {
    output[index] = 0.0f;
    return;
  }
  const uint32_t input_index =
      ((batch * input_channels + channel) * input_height + input_y) *
          input_width +
      input_x;
  output[index] = input[input_offset + input_index];
}

template <bool WITH_BIAS>
__device__ __forceinline__ void nhwc_to_nchw_batch_norm(
    const float* input,
    const float* convolution_bias,
    const float* scale_and_bias,
    const float* mean_and_stddev,
    float* output,
    uint64_t input_offset,
    uint64_t convolution_bias_offset,
    uint64_t scale_and_bias_offset,
    uint64_t mean_and_stddev_offset,
    uint64_t batch,
    uint64_t channels,
    uint64_t spatial,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = batch * channels * spatial;
  if (output_index >= elements) {
    return;
  }
  const uint64_t spatial_index = output_index % spatial;
  const uint64_t channel = (output_index / spatial) % channels;
  const uint64_t batch_index = output_index / (channels * spatial);
  const uint64_t input_index =
      (batch_index * spatial + spatial_index) * channels + channel;
  float value = input[input_offset + input_index];
  if constexpr (WITH_BIAS) {
    value = __fadd_rn(
        value, convolution_bias[convolution_bias_offset + channel]);
  }
  const float scale = scale_and_bias[scale_and_bias_offset + channel];
  const float bias =
      scale_and_bias[scale_and_bias_offset + channels + channel];
  const float mean = mean_and_stddev[mean_and_stddev_offset + channel];
  const float stddev =
      mean_and_stddev[mean_and_stddev_offset + channels + channel];
  const float centered = __fsub_rn(value, mean);
  const float normalized = __fdiv_rn(centered, stddev);
  const float scaled = __fmul_rn(normalized, scale);
  const float normalized_output = __fadd_rn(scaled, bias);
  if (activation == 0) {
    output[output_index] = normalized_output;
    return;
  }
  if (activation == 2) {
    output[output_index] = fmaxf(normalized_output, 0.0f);
    return;
  }
  if (activation == 3) {
    const float exponential = expf(-normalized_output);
    const float denominator = __fadd_rn(1.0f, exponential);
    const float gate = __fdiv_rn(1.0f, denominator);
    output[output_index] = __fmul_rn(normalized_output, gate);
    return;
  }
  if (activation == 4) {
    const float divided = __fdiv_rn(normalized_output, alpha);
    const float activated = erff(divided);
    const float shifted = __fadd_rn(activated, beta);
    const float product = __fmul_rn(normalized_output, shifted);
    output[output_index] = __fmul_rn(product, gamma);
    return;
  }
  const float gate_scaled = __fmaf_rn(normalized_output, alpha, 0.0f);
  const float gate_shifted = __fmaf_rn(gate_scaled, 1.0f, beta);
  const float lower_bounded = fmaxf(gate_shifted, 0.0f);
  const float gate = fminf(lower_bounded, 1.0f);
  output[output_index] = __fmul_rn(normalized_output, gate);
}

extern "C" __global__ void nhwc_to_nchw_batch_norm_f32(
    const float* input,
    const float* scale_and_bias,
    const float* mean_and_stddev,
    float* output,
    uint64_t input_offset,
    uint64_t scale_and_bias_offset,
    uint64_t mean_and_stddev_offset,
    uint64_t batch,
    uint64_t channels,
    uint64_t spatial,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  nhwc_to_nchw_batch_norm<false>(
      input,
      nullptr,
      scale_and_bias,
      mean_and_stddev,
      output,
      input_offset,
      0,
      scale_and_bias_offset,
      mean_and_stddev_offset,
      batch,
      channels,
      spatial,
      activation,
      alpha,
      beta,
      gamma);
}

extern "C" __global__ void nhwc_to_nchw_bias_batch_norm_f32(
    const float* input,
    const float* convolution_bias,
    const float* scale_and_bias,
    const float* mean_and_stddev,
    float* output,
    uint64_t input_offset,
    uint64_t convolution_bias_offset,
    uint64_t scale_and_bias_offset,
    uint64_t mean_and_stddev_offset,
    uint64_t batch,
    uint64_t channels,
    uint64_t spatial,
    uint32_t activation,
    float alpha,
    float beta,
    float gamma) {
  nhwc_to_nchw_batch_norm<true>(
      input,
      convolution_bias,
      scale_and_bias,
      mean_and_stddev,
      output,
      input_offset,
      convolution_bias_offset,
      scale_and_bias_offset,
      mean_and_stddev_offset,
      batch,
      channels,
      spatial,
      activation,
      alpha,
      beta,
      gamma);
}

template <bool WITH_BIAS>
__device__ __forceinline__ void nhwc_to_nchw_activation(
    const float* input,
    const float* bias,
    float* output,
    uint64_t input_offset,
    uint64_t bias_offset,
    uint64_t element_count,
    uint64_t channels,
    uint64_t spatial,
    uint32_t activation,
    float divisor,
    float offset,
    float scale) {
  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= element_count) {
    return;
  }
  const uint64_t spatial_index = output_index % spatial;
  const uint64_t channel = (output_index / spatial) % channels;
  const uint64_t batch_index = output_index / (channels * spatial);
  const uint64_t input_index =
      (batch_index * spatial + spatial_index) * channels + channel;
  float value = input[input_offset + input_index];
  if constexpr (WITH_BIAS) {
    value = __fadd_rn(value, bias[bias_offset + channel]);
  }
  if (activation == 0) {
    output[output_index] = fmaxf(value, 0.0f);
    return;
  }
  const float divided = __fdiv_rn(value, divisor);
  const float activated = erff(divided);
  const float shifted = __fadd_rn(activated, offset);
  const float product = __fmul_rn(value, shifted);
  output[output_index] = __fmul_rn(product, scale);
}

extern "C" __global__ void nhwc_to_nchw_activation_f32(
    const float* input,
    float* output,
    uint64_t input_offset,
    uint64_t element_count,
    uint64_t channels,
    uint64_t spatial,
    uint32_t activation,
    float divisor,
    float offset,
    float scale) {
  nhwc_to_nchw_activation<false>(
      input,
      nullptr,
      output,
      input_offset,
      0,
      element_count,
      channels,
      spatial,
      activation,
      divisor,
      offset,
      scale);
}

extern "C" __global__ void nhwc_to_nchw_bias_activation_f32(
    const float* input,
    const float* bias,
    float* output,
    uint64_t input_offset,
    uint64_t bias_offset,
    uint64_t element_count,
    uint64_t channels,
    uint64_t spatial,
    uint32_t activation,
    float divisor,
    float offset,
    float scale) {
  nhwc_to_nchw_activation<true>(
      input,
      bias,
      output,
      input_offset,
      bias_offset,
      element_count,
      channels,
      spatial,
      activation,
      divisor,
      offset,
      scale);
}
