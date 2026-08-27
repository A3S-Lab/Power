#ifndef A3S_POWER_CUDA_FAST_DIVIDE_CUH
#define A3S_POWER_CUDA_FAST_DIVIDE_CUH

#include <stdint.h>

__device__ __forceinline__ uint32_t fast_divide_u32(
    uint32_t numerator,
    uint32_t multiplier,
    uint32_t shift) {
  const uint32_t high_product = __umulhi(numerator, multiplier);
  return static_cast<uint32_t>(
      (static_cast<uint64_t>(numerator) + high_product) >> shift);
}

#endif
