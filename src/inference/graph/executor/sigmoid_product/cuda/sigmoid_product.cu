#include <math.h>
#include <stdint.h>

#include "../../cuda_fast_divide.cuh"

__device__ __forceinline__ float sigmoid_rn(const float value) {
  const float exponential = expf(-value);
  const float denominator = __fadd_rn(1.0f, exponential);
  return __fdiv_rn(1.0f, denominator);
}

extern "C" __global__ void sigmoid_product_f32(
    const float *left,
    const float *right,
    float *output,
    const uint64_t left_offset,
    const uint64_t right_offset,
    const uint32_t element_count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float left_gate = sigmoid_rn(left[left_offset + index]);
  const float right_gate = sigmoid_rn(right[right_offset + index]);
  output[index] = __fmul_rn(left_gate, right_gate);
}

extern "C" __global__ void sigmoid_mul_f32(
    const float *input,
    const float *multiplier,
    float *output,
    const uint64_t input_offset,
    const uint64_t multiplier_offset,
    const uint32_t element_count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  output[index] = __fmul_rn(
      sigmoid_rn(input[input_offset + index]),
      multiplier[multiplier_offset + index]);
}

extern "C" __global__ void sigmoid_mul_nchw_per_channel_f32(
    const float *input,
    const float *multiplier,
    float *output,
    const uint64_t input_offset,
    const uint64_t multiplier_offset,
    const uint32_t element_count,
    const uint32_t spatial_multiplier,
    const uint32_t spatial_shift,
    const uint32_t spatial,
    const uint32_t channels_multiplier,
    const uint32_t channels_shift,
    const uint32_t channels) {
  (void)spatial;
  (void)channels_multiplier;
  (void)channels_shift;
  (void)channels;
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const uint32_t multiplier_index =
      fast_divide_u32(index, spatial_multiplier, spatial_shift);
  output[index] = __fmul_rn(
      sigmoid_rn(input[input_offset + index]),
      multiplier[multiplier_offset + multiplier_index]);
}

extern "C" __global__ void sigmoid_mul_nchw_per_spatial_position_f32(
    const float *input,
    const float *multiplier,
    float *output,
    const uint64_t input_offset,
    const uint64_t multiplier_offset,
    const uint32_t element_count,
    const uint32_t spatial_multiplier,
    const uint32_t spatial_shift,
    const uint32_t spatial,
    const uint32_t channels_multiplier,
    const uint32_t channels_shift,
    const uint32_t channels) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const uint32_t channel_batch =
      fast_divide_u32(index, spatial_multiplier, spatial_shift);
  const uint32_t batch =
      fast_divide_u32(channel_batch, channels_multiplier, channels_shift);
  const uint32_t spatial_position = index - channel_batch * spatial;
  const uint32_t multiplier_index = batch * spatial + spatial_position;
  (void)channels;
  output[index] = __fmul_rn(
      sigmoid_rn(input[input_offset + index]),
      multiplier[multiplier_offset + multiplier_index]);
}
