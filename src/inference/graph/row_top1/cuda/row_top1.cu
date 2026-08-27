#include <math.h>
#include <stdint.h>

extern "C" __global__ void row_top1_last_finite_f32(
    const float *input,
    float *output,
    const uint64_t input_offset,
    const uint32_t rows,
    const uint32_t classes) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float best_score = -INFINITY;
    uint32_t best_index = 0;
    uint32_t all_finite = 1;
    const uint64_t row_offset = input_offset + (uint64_t)row * (uint64_t)classes;
    for (uint32_t class_index = threadIdx.x; class_index < classes;
         class_index += blockDim.x) {
        const float score = input[row_offset + class_index];
        all_finite &= (uint32_t)isfinite(score);
        if (score > best_score || (score == best_score && class_index > best_index)) {
            best_score = score;
            best_index = class_index;
        }
    }

    extern __shared__ unsigned char shared_storage[];
    float *scores = reinterpret_cast<float *>(shared_storage);
    uint32_t *indices = reinterpret_cast<uint32_t *>(scores + blockDim.x);
    uint32_t *finite = indices + blockDim.x;
    scores[threadIdx.x] = best_score;
    indices[threadIdx.x] = best_index;
    finite[threadIdx.x] = all_finite;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float candidate_score = scores[threadIdx.x + stride];
            const uint32_t candidate_index = indices[threadIdx.x + stride];
            if (candidate_score > scores[threadIdx.x] ||
                (candidate_score == scores[threadIdx.x] &&
                 candidate_index > indices[threadIdx.x])) {
                scores[threadIdx.x] = candidate_score;
                indices[threadIdx.x] = candidate_index;
            }
            finite[threadIdx.x] &= finite[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const uint64_t output_offset = (uint64_t)row * 3;
        output[output_offset] = (float)indices[0];
        output[output_offset + 1] = scores[0];
        output[output_offset + 2] = finite[0] ? 1.0f : 0.0f;
    }
}
