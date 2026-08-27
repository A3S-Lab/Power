#ifndef A3S_POWER_DEPTHWISE_CONTIGUOUS_CUH
#define A3S_POWER_DEPTHWISE_CONTIGUOUS_CUH

#include <stdint.h>
#include "../../cuda_fast_divide.cuh"

__device__ __forceinline__ float depthwise_accumulate_contiguous_u32(
    const float* input,
    const float* kernel,
    uint32_t batch_index,
    uint32_t channel,
    uint32_t output_y,
    uint32_t output_x,
    uint32_t channels,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation,
    uint32_t pad_top,
    uint32_t pad_left) {
  float accumulator = 0.0f;
  bool first = true;
  for (uint32_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
    const uint32_t padded_y =
        output_y * stride_height + kernel_y * dilation;
    for (uint32_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
      const uint32_t padded_x =
          output_x * stride_width + kernel_x * dilation;
      const uint32_t kernel_index =
          (channel * kernel_height + kernel_y) * kernel_width + kernel_x;
      float input_value = 0.0f;
      if (padded_y >= pad_top && padded_x >= pad_left) {
        const uint32_t input_y = padded_y - pad_top;
        const uint32_t input_x = padded_x - pad_left;
        if (input_y < input_height && input_x < input_width) {
          const uint32_t input_index =
              ((batch_index * channels + channel) * input_height + input_y) *
                  input_width +
              input_x;
          input_value = input[input_index];
        }
      }
      const float product = __fmul_rn(input_value, kernel[kernel_index]);
      if (first) {
        accumulator = product;
        first = false;
      } else {
        accumulator = __fadd_rn(accumulator, product);
      }
    }
  }
  return accumulator;
}

__device__ __forceinline__ void decode_contiguous_u32_index(
    uint32_t index,
    uint32_t output_width_multiplier,
    uint32_t output_width_shift,
    uint32_t output_width,
    uint32_t spatial_multiplier,
    uint32_t spatial_shift,
    uint32_t spatial,
    uint32_t channels_multiplier,
    uint32_t channels_shift,
    uint32_t channels,
    uint32_t& output_x,
    uint32_t& output_y,
    uint32_t& channel,
    uint32_t& batch_index) {
  const uint32_t channel_batch =
      fast_divide_u32(index, spatial_multiplier, spatial_shift);
  const uint32_t spatial_index = index - channel_batch * spatial;
  output_y = fast_divide_u32(
      spatial_index, output_width_multiplier, output_width_shift);
  output_x = spatial_index - output_y * output_width;
  batch_index =
      fast_divide_u32(channel_batch, channels_multiplier, channels_shift);
  channel = channel_batch - batch_index * channels;
}

#endif
