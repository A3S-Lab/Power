#include <math.h>
#include <stdint.h>
#include "../../cuda_fast_divide.cuh"

extern "C" __global__ void last_axis_bias_in_place_f32(
    float *output,
    const float *bias,
    const uint64_t bias_offset,
    const uint32_t element_count,
    const uint32_t feature_multiplier,
    const uint32_t feature_shift,
    const uint32_t features) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const uint32_t row =
      fast_divide_u32(index, feature_multiplier, feature_shift);
  const uint32_t feature = index - row * features;
  output[index] =
      __fadd_rn(output[index], bias[bias_offset + feature]);
}

extern "C" __global__ void last_axis_bias_swish_in_place_f32(
    float *output,
    const float *bias,
    const uint64_t bias_offset,
    const uint32_t element_count,
    const uint32_t feature_multiplier,
    const uint32_t feature_shift,
    const uint32_t features) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const uint32_t row =
      fast_divide_u32(index, feature_multiplier, feature_shift);
  const uint32_t feature = index - row * features;
  const float value =
      __fadd_rn(output[index], bias[bias_offset + feature]);
  const float exponential = expf(-value);
  const float denominator = __fadd_rn(1.0f, exponential);
  const float gate = __fdiv_rn(1.0f, denominator);
  output[index] = __fmul_rn(value, gate);
}
