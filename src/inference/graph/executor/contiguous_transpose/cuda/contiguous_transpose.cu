#include <stdint.h>

constexpr uint32_t TILE = 32;
constexpr uint32_t BLOCK_ROWS = 8;

extern "C" __global__ void contiguous_last_two_transpose_u32_f32(
    const float* input,
    float* output,
    uint64_t input_offset,
    uint32_t rows,
    uint32_t columns) {
  __shared__ float tile[TILE][TILE + 1];

  const uint32_t source_x = blockIdx.x * TILE + threadIdx.x;
  const uint32_t source_y = blockIdx.y * TILE + threadIdx.y;
  const uint64_t matrix_offset =
      static_cast<uint64_t>(blockIdx.z) * rows * columns;
  for (uint32_t offset = 0; offset < TILE; offset += BLOCK_ROWS) {
    if (source_x < rows && source_y + offset < columns) {
      tile[threadIdx.y + offset][threadIdx.x] =
          input[input_offset + matrix_offset +
                static_cast<uint64_t>(source_y + offset) * rows + source_x];
    }
  }
  __syncthreads();

  const uint32_t output_x = blockIdx.y * TILE + threadIdx.x;
  const uint32_t output_y = blockIdx.x * TILE + threadIdx.y;
  for (uint32_t offset = 0; offset < TILE; offset += BLOCK_ROWS) {
    if (output_x < columns && output_y + offset < rows) {
      output[matrix_offset +
             static_cast<uint64_t>(output_y + offset) * columns + output_x] =
          tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}
