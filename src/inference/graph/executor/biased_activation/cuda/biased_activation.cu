#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float bounded_hard_sigmoid(
    const float value,
    const float alpha,
    const float beta) {
    const float scaled = __fmaf_rn(value, alpha, 0.0f);
    const float shifted = __fmaf_rn(scaled, 1.0f, beta);
    const float lower_bounded = fmaxf(shifted, 0.0f);
    return fminf(lower_bounded, 1.0f);
}

extern "C" __global__ void channel_bias_relu_f32(
    const float *input,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t bias_offset,
    const uint64_t element_count,
    const uint32_t channels,
    const uint32_t spatial_elements,
    const float divisor,
    const float offset,
    const float scale) {
    (void)divisor;
    (void)offset;
    (void)scale;
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const uint32_t flat_index = (uint32_t)index;
        const uint32_t channel = (flat_index / spatial_elements) % channels;
        const float biased =
            __fadd_rn(input[input_offset + index], bias[bias_offset + channel]);
        output[index] = fmaxf(biased, 0.0f);
    }
}

extern "C" __global__ void channel_bias_gelu_erf_f32(
    const float *input,
    const float *bias,
    float *output,
    const uint64_t input_offset,
    const uint64_t bias_offset,
    const uint64_t element_count,
    const uint32_t channels,
    const uint32_t spatial_elements,
    const float divisor,
    const float offset,
    const float scale) {
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const uint32_t flat_index = (uint32_t)index;
        const uint32_t channel = (flat_index / spatial_elements) % channels;
        const float value =
            __fadd_rn(input[input_offset + index], bias[bias_offset + channel]);
        const float divided = __fdiv_rn(value, divisor);
        const float activated = erff(divided);
        const float shifted = __fadd_rn(activated, offset);
        const float product = __fmul_rn(value, shifted);
        output[index] = __fmul_rn(product, scale);
    }
}

extern "C" __global__ void channel_bias_gated_hard_sigmoid_mul_f32(
    const float *multiplicand,
    const float *gate,
    const float *bias,
    float *output,
    const uint64_t multiplicand_offset,
    const uint64_t gate_offset,
    const uint64_t bias_offset,
    const uint64_t element_count,
    const uint32_t channels,
    const uint32_t spatial_elements,
    const float alpha,
    const float beta) {
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const uint32_t flat_index = (uint32_t)index;
        const uint32_t channel = (flat_index / spatial_elements) % channels;
        const float biased_gate =
            __fadd_rn(gate[gate_offset + index], bias[bias_offset + channel]);
        const float bounded = bounded_hard_sigmoid(biased_gate, alpha, beta);
        output[index] =
            __fmul_rn(multiplicand[multiplicand_offset + index], bounded);
    }
}

extern "C" __global__ void channel_bias_gated_hard_sigmoid_channel_mul_f32(
    const float *multiplicand,
    const float *gate,
    const float *bias,
    float *output,
    const uint64_t multiplicand_offset,
    const uint64_t gate_offset,
    const uint64_t bias_offset,
    const uint64_t element_count,
    const uint32_t channels,
    const uint32_t spatial_elements,
    const float alpha,
    const float beta) {
    const uint64_t start =
        (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t step = (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    for (uint64_t index = start; index < element_count; index += step) {
        const uint32_t flat_index = (uint32_t)index;
        const uint32_t gate_index = flat_index / spatial_elements;
        const uint32_t channel = gate_index % channels;
        const float biased_gate = __fadd_rn(
            gate[gate_offset + gate_index],
            bias[bias_offset + channel]);
        const float bounded = bounded_hard_sigmoid(biased_gate, alpha, beta);
        output[index] =
            __fmul_rn(multiplicand[multiplicand_offset + index], bounded);
    }
}
