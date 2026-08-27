#include <stdint.h>

__device__ __forceinline__ float depthwise_accumulate(
    const float* input,
    const float* kernel,
    uint64_t batch_index,
    uint64_t channel,
    uint64_t output_y,
    uint64_t output_x,
    uint64_t kernel_height,
    uint64_t kernel_width,
    uint64_t stride_height,
    uint64_t stride_width,
    uint64_t dilation,
    uint64_t pad_top,
    uint64_t pad_left,
    uint64_t input_height,
    uint64_t input_width,
    uint64_t input_offset,
    uint64_t input_stride_batch,
    uint64_t input_stride_channel,
    uint64_t input_stride_height,
    uint64_t input_stride_width,
    uint64_t kernel_offset,
    uint64_t kernel_stride_channel,
    uint64_t kernel_stride_height,
    uint64_t kernel_stride_width) {
  float accumulator = 0.0f;
  bool first = true;
  for (uint64_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
    const uint64_t padded_y =
        output_y * stride_height + kernel_y * dilation;
    for (uint64_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
      const uint64_t padded_x =
          output_x * stride_width + kernel_x * dilation;
      const uint64_t kernel_index =
          kernel_offset + channel * kernel_stride_channel +
          kernel_y * kernel_stride_height + kernel_x * kernel_stride_width;
      float input_value = 0.0f;
      if (padded_y >= pad_top && padded_x >= pad_left) {
        const uint64_t input_y = padded_y - pad_top;
        const uint64_t input_x = padded_x - pad_left;
        if (input_y < input_height && input_x < input_width) {
          const uint64_t input_index =
              input_offset + batch_index * input_stride_batch +
              channel * input_stride_channel + input_y * input_stride_height +
              input_x * input_stride_width;
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

extern "C" __global__ void depthwise_conv2d_f32(
    const float* input,
    const float* kernel,
    float* output,
    uint64_t batch,
    uint64_t channels,
    uint64_t input_height,
    uint64_t input_width,
    uint64_t output_height,
    uint64_t output_width,
    uint64_t kernel_height,
    uint64_t kernel_width,
    uint64_t stride_height,
    uint64_t stride_width,
    uint64_t dilation,
    uint64_t pad_top,
    uint64_t pad_left,
    uint64_t input_offset,
    uint64_t input_stride_batch,
    uint64_t input_stride_channel,
    uint64_t input_stride_height,
    uint64_t input_stride_width,
    uint64_t kernel_offset,
    uint64_t kernel_stride_channel,
    uint64_t kernel_stride_height,
    uint64_t kernel_stride_width,
    uint64_t bias_offset,
    uint64_t bias_stride) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t spatial = output_height * output_width;
  const uint64_t elements = batch * channels * spatial;
  if (index >= elements) {
    return;
  }
  const uint64_t output_x = index % output_width;
  const uint64_t output_y = (index / output_width) % output_height;
  const uint64_t channel = (index / spatial) % channels;
  const uint64_t batch_index = index / (channels * spatial);
  output[index] = depthwise_accumulate(
      input, kernel, batch_index, channel, output_y, output_x, kernel_height,
      kernel_width, stride_height, stride_width, dilation, pad_top, pad_left,
      input_height, input_width, input_offset, input_stride_batch,
      input_stride_channel, input_stride_height, input_stride_width,
      kernel_offset, kernel_stride_channel,
      kernel_stride_height, kernel_stride_width);
}

extern "C" __global__ void depthwise_conv2d_bias_f32(
    const float* input,
    const float* kernel,
    const float* bias,
    float* output,
    uint64_t batch,
    uint64_t channels,
    uint64_t input_height,
    uint64_t input_width,
    uint64_t output_height,
    uint64_t output_width,
    uint64_t kernel_height,
    uint64_t kernel_width,
    uint64_t stride_height,
    uint64_t stride_width,
    uint64_t dilation,
    uint64_t pad_top,
    uint64_t pad_left,
    uint64_t input_offset,
    uint64_t input_stride_batch,
    uint64_t input_stride_channel,
    uint64_t input_stride_height,
    uint64_t input_stride_width,
    uint64_t kernel_offset,
    uint64_t kernel_stride_channel,
    uint64_t kernel_stride_height,
    uint64_t kernel_stride_width,
    uint64_t bias_offset,
    uint64_t bias_stride) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t spatial = output_height * output_width;
  const uint64_t elements = batch * channels * spatial;
  if (index >= elements) {
    return;
  }
  const uint64_t output_x = index % output_width;
  const uint64_t output_y = (index / output_width) % output_height;
  const uint64_t channel = (index / spatial) % channels;
  const uint64_t batch_index = index / (channels * spatial);
  const float convolution = depthwise_accumulate(
      input, kernel, batch_index, channel, output_y, output_x, kernel_height,
      kernel_width, stride_height, stride_width, dilation, pad_top, pad_left,
      input_height, input_width, input_offset, input_stride_batch,
      input_stride_channel, input_stride_height, input_stride_width,
      kernel_offset, kernel_stride_channel,
      kernel_stride_height, kernel_stride_width);
  output[index] = __fadd_rn(convolution, bias[bias_offset + channel * bias_stride]);
}

#include "contiguous.cuh"

extern "C" __global__ void depthwise_conv2d_contiguous_u32_f32(
    const float* input,
    const float* kernel,
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
    uint32_t channels_divisor) {
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
  output[index] = depthwise_accumulate_contiguous_u32(
      input, kernel, batch_index, channel, output_y, output_x, channels,
      input_height, input_width, kernel_height, kernel_width, stride_height,
      stride_width, dilation, pad_top, pad_left);
}

extern "C" __global__ void depthwise_conv2d_bias_contiguous_u32_f32(
    const float* input,
    const float* kernel,
    const float* bias,
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
    uint32_t channels_divisor) {
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
  output[index] = __fadd_rn(convolution, bias[channel]);
}
