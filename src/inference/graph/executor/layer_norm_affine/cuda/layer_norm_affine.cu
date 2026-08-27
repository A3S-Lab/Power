#include <math.h>
#include <stdint.h>

extern "C" __global__ void layer_norm_full_f32(
    const float *input,
    const float *scale,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t scale_offset,
    const uint64_t bias_offset,
    const uint64_t element_count,
    const uint32_t features,
    const float mean_scale,
    const float epsilon) {
    const uint64_t row = (uint64_t)blockIdx.x;
    const uint32_t thread = threadIdx.x;
    const uint64_t row_start = row * (uint64_t)features;
    if (row_start >= element_count) {
        return;
    }

    __shared__ float partial[1024];
    __shared__ float row_mean;
    float sum = 0.0f;
    for (uint32_t feature = thread; feature < features; feature += blockDim.x) {
        sum = __fadd_rn(sum, input[input_offset + row_start + feature]);
    }
    partial[thread] = sum;
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (thread < stride) {
            partial[thread] =
                __fadd_rn(partial[thread], partial[thread + stride]);
        }
    }
    __syncthreads();
    if (thread == 0) {
        row_mean = __fmul_rn(partial[0], mean_scale);
    }
    // Publish the mean through independent shared storage before reusing the
    // reduction buffer. Without this barrier, thread zero can overwrite
    // partial[0] with its square sum while another warp is still loading the
    // mean, making identical inference inputs history-dependent.
    __syncthreads();
    const float mean = row_mean;

    float square_sum = 0.0f;
    for (uint32_t feature = thread; feature < features; feature += blockDim.x) {
        const float centered =
            __fsub_rn(input[input_offset + row_start + feature], mean);
        square_sum = __fadd_rn(square_sum, __fmul_rn(centered, centered));
    }
    partial[thread] = square_sum;
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (thread < stride) {
            partial[thread] =
                __fadd_rn(partial[thread], partial[thread + stride]);
        }
    }
    __syncthreads();
    const float variance = __fmul_rn(partial[0], mean_scale);
    const float denominator = sqrtf(__fadd_rn(variance, epsilon));

    for (uint32_t feature = thread; feature < features; feature += blockDim.x) {
        const uint64_t index = row_start + feature;
        const float centered = __fsub_rn(input[input_offset + index], mean);
        const float normalized = __fdiv_rn(centered, denominator);
        const float scaled =
            __fmul_rn(normalized, scale[scale_offset + feature]);
        output[index] = __fadd_rn(scaled, bias[bias_offset + feature]);
    }
}

extern "C" __global__ void layer_norm_affine_tail_f32(
    const float *centered,
    const float *variance,
    const float *scale,
    const float *bias,
    float *output,
    const uint64_t centered_offset,
    const uint64_t variance_offset,
    const uint64_t scale_offset,
    const uint64_t bias_offset,
    const uint64_t element_count,
    const uint32_t features,
    const float epsilon) {
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const uint32_t flat_index = (uint32_t)index;
        const uint32_t feature = flat_index % features;
        const uint32_t row = flat_index / features;
        const float shifted_variance =
            __fadd_rn(variance[variance_offset + row], epsilon);
        const float denominator = sqrtf(shifted_variance);
        const float normalized =
            __fdiv_rn(centered[centered_offset + index], denominator);
        const float scaled =
            __fmul_rn(normalized, scale[scale_offset + feature]);
        output[index] = __fadd_rn(scaled, bias[bias_offset + feature]);
    }
}
