#include <math.h>
#include <stdint.h>

template <bool WITH_BIAS>
__device__ __forceinline__ void project_row_softmax_top1(
    const float *input,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t bias_offset,
    const uint32_t rows,
    const uint32_t classes) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    constexpr uint32_t REDUCTION_LANES = 1024;
    const uint64_t row_offset = input_offset + (uint64_t)row * (uint64_t)classes;
    extern __shared__ unsigned char shared_storage[];
    float *scores = reinterpret_cast<float *>(shared_storage);
    uint32_t *indices = reinterpret_cast<uint32_t *>(scores + REDUCTION_LANES);
    uint32_t *finite = indices + REDUCTION_LANES;
    float *sums = reinterpret_cast<float *>(finite + REDUCTION_LANES);
    for (uint32_t lane = threadIdx.x; lane < REDUCTION_LANES;
         lane += blockDim.x) {
        float best_score = -INFINITY;
        uint32_t best_index = 0;
        uint32_t all_finite = 1;
        for (uint32_t class_index = lane; class_index < classes;
             class_index += REDUCTION_LANES) {
            float score = input[row_offset + class_index];
            if (WITH_BIAS) {
                score += bias[bias_offset + class_index];
            }
            all_finite &= (uint32_t)isfinite(score);
            if (score > best_score ||
                (score == best_score && class_index > best_index)) {
                best_score = score;
                best_index = class_index;
            }
        }
        scores[lane] = best_score;
        indices[lane] = best_index;
        finite[lane] = all_finite;
    }
    __syncthreads();

    for (uint32_t stride = REDUCTION_LANES / 2; stride >= 32; stride >>= 1) {
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
    if (threadIdx.x < 32) {
        float local_score = scores[threadIdx.x];
        uint32_t local_index = indices[threadIdx.x];
        uint32_t local_finite = finite[threadIdx.x];
        for (uint32_t stride = 16; stride > 0; stride >>= 1) {
            const float candidate_score =
                __shfl_down_sync(0xffffffff, local_score, stride);
            const uint32_t candidate_index =
                __shfl_down_sync(0xffffffff, local_index, stride);
            const uint32_t candidate_finite =
                __shfl_down_sync(0xffffffff, local_finite, stride);
            if (threadIdx.x < stride) {
                if (candidate_score > local_score ||
                    (candidate_score == local_score &&
                     candidate_index > local_index)) {
                    local_score = candidate_score;
                    local_index = candidate_index;
                }
                local_finite &= candidate_finite;
            }
        }
        if (threadIdx.x == 0) {
            scores[0] = local_score;
            indices[0] = local_index;
            finite[0] = local_finite;
        }
    }
    __syncthreads();

    const float maximum = scores[0];
    for (uint32_t lane = threadIdx.x; lane < REDUCTION_LANES;
         lane += blockDim.x) {
        float sum = 0.0f;
        if (finite[0]) {
            for (uint32_t class_index = lane; class_index < classes;
                 class_index += REDUCTION_LANES) {
                float score = input[row_offset + class_index];
                if (WITH_BIAS) {
                    score += bias[bias_offset + class_index];
                }
                sum += expf(score - maximum);
            }
        }
        sums[lane] = sum;
    }
    __syncthreads();
    for (uint32_t stride = REDUCTION_LANES / 2; stride >= 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float total_sum = 0.0f;
    if (threadIdx.x < 32) {
        total_sum = sums[threadIdx.x];
        for (uint32_t stride = 16; stride > 0; stride >>= 1) {
            const float candidate =
                __shfl_down_sync(0xffffffff, total_sum, stride);
            if (threadIdx.x < stride) {
                total_sum = __fadd_rn(total_sum, candidate);
            }
        }
    }

    if (threadIdx.x == 0) {
        const uint64_t output_offset = (uint64_t)row * 3;
        output[output_offset] = (float)indices[0];
        output[output_offset + 1] = finite[0] ? 1.0f / total_sum : 0.0f;
        output[output_offset + 2] = finite[0] ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void row_softmax_top1_last_finite_f32(
    const float *input,
    float *output,
    const uint64_t input_offset,
    const uint32_t rows,
    const uint32_t classes) {
    project_row_softmax_top1<false>(
        input, input, output, input_offset, 0, rows, classes);
}

extern "C" __global__ void row_bias_softmax_top1_last_finite_f32(
    const float *input,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t bias_offset,
    const uint32_t rows,
    const uint32_t classes) {
    project_row_softmax_top1<true>(
        input, bias, output, input_offset, bias_offset, rows, classes);
}
