#include "dnnl.hpp"
#include "dnnl_config.h"
#include "activation_arena.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using dnnl::algorithm;
using dnnl::convolution_forward;
using dnnl::engine;
using dnnl::memory;
using dnnl::post_ops;
using dnnl::primitive;
using dnnl::primitive_attr;
using dnnl::reorder;
using dnnl::stream;

struct NativeBlock {
    std::uint64_t input_channels;
    std::uint64_t output_channels;
    std::uint64_t groups;
    std::uint64_t kernel_height;
    std::uint64_t kernel_width;
    std::uint64_t stride_height;
    std::uint64_t stride_width;
    std::uint64_t dilation_height;
    std::uint64_t dilation_width;
    std::uint32_t padding_kind;
    std::uint64_t padding_top;
    std::uint64_t padding_left;
    std::uint64_t padding_bottom;
    std::uint64_t padding_right;
    std::uint32_t activation_kind;
    std::int64_t residual_source;
    const float *weights;
    std::uint64_t weight_count;
    const float *bias;
    std::uint64_t bias_count;
};

struct Block {
    std::int64_t input_channels;
    std::int64_t output_channels;
    std::int64_t groups;
    std::int64_t kernel_height;
    std::int64_t kernel_width;
    std::int64_t stride_height;
    std::int64_t stride_width;
    std::int64_t dilation_height;
    std::int64_t dilation_width;
    std::uint32_t padding_kind;
    std::int64_t padding_top;
    std::int64_t padding_left;
    std::int64_t padding_bottom;
    std::int64_t padding_right;
    std::uint32_t activation_kind;
    std::int64_t residual_source;
    std::vector<float> weights;
    std::vector<float> bias;
};

using ShapeKey = std::array<std::int64_t, 4>;

memory::data_type internal_data_type(std::uint32_t kind) {
    switch (kind) {
    case 0:
        return memory::data_type::f32;
    case 1:
        return memory::data_type::bf16;
    default:
        throw std::invalid_argument(
                "optimized CPU graph precision kind is unsupported");
    }
}

struct ShapeHash {
    std::size_t operator()(const ShapeKey &shape) const noexcept {
        std::size_t hash = 0xcbf29ce484222325ULL;
        for (const auto value : shape) {
            hash ^= static_cast<std::size_t>(value);
            hash *= 0x100000001b3ULL;
        }
        return hash;
    }
};

void configure_threading() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
}

std::int64_t checked_i64(std::uint64_t value, const char *label) {
    if (value == 0
            || value
                    > static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(std::string(label)
                + " must fit a positive signed 64-bit dimension");
    }
    return static_cast<std::int64_t>(value);
}

std::uint64_t checked_product(
        std::initializer_list<std::uint64_t> values, const char *label) {
    std::uint64_t product = 1;
    for (const auto value : values) {
        if (value == 0
                || product > std::numeric_limits<std::uint64_t>::max() / value) {
            throw std::invalid_argument(
                    std::string(label) + " element count overflowed");
        }
        product *= value;
    }
    return product;
}

std::size_t checked_size(std::uint64_t value, const char *label) {
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
                std::string(label) + " exceeds the host size type");
    }
    return static_cast<std::size_t>(value);
}

std::int64_t output_dimension(std::int64_t input, std::int64_t kernel,
        std::int64_t stride, std::int64_t dilation, std::int64_t before,
        std::int64_t after) {
    const auto effective = dilation * (kernel - 1) + 1;
    const auto padded = input + before + after;
    if (padded < effective) {
        throw std::invalid_argument(
                "optimized convolution kernel exceeds its input");
    }
    return (padded - effective) / stride + 1;
}

std::pair<std::int64_t, std::int64_t> same_upper_padding(
        std::int64_t input, std::int64_t kernel, std::int64_t stride,
        std::int64_t dilation) {
    const auto output = (input + stride - 1) / stride;
    const auto effective = dilation * (kernel - 1) + 1;
    const auto total
            = std::max<std::int64_t>(0, (output - 1) * stride + effective - input);
    return {total / 2, total - total / 2};
}

std::vector<Block> copy_blocks(
        const NativeBlock *blocks, std::size_t block_count) {
    if (blocks == nullptr || block_count < 2) {
        throw std::invalid_argument(
                "optimized CPU graph segment requires at least two blocks");
    }
    std::vector<Block> result;
    result.reserve(block_count);
    std::int64_t preceding_channels = 0;
    for (std::size_t index = 0; index < block_count; ++index) {
        const auto &source = blocks[index];
        Block block {
                checked_i64(source.input_channels, "input channels"),
                checked_i64(source.output_channels, "output channels"),
                checked_i64(source.groups, "groups"),
                checked_i64(source.kernel_height, "kernel height"),
                checked_i64(source.kernel_width, "kernel width"),
                checked_i64(source.stride_height, "stride height"),
                checked_i64(source.stride_width, "stride width"),
                checked_i64(source.dilation_height, "dilation height"),
                checked_i64(source.dilation_width, "dilation width"),
                source.padding_kind,
                static_cast<std::int64_t>(source.padding_top),
                static_cast<std::int64_t>(source.padding_left),
                static_cast<std::int64_t>(source.padding_bottom),
                static_cast<std::int64_t>(source.padding_right),
                source.activation_kind,
                source.residual_source,
                {},
                {}};
        if (block.padding_kind > 1 || block.activation_kind > 2
                || block.input_channels % block.groups != 0
                || block.output_channels % block.groups != 0
                || block.residual_source < -2
                || block.residual_source >= static_cast<std::int64_t>(index)
                || (block.residual_source == -2 && index == 0)
                || (index != 0
                    && block.input_channels != preceding_channels)) {
            throw std::invalid_argument(
                    "optimized CPU graph segment has invalid block topology");
        }
        const auto expected_weights = checked_product(
                {source.output_channels,
                        source.input_channels / source.groups,
                        source.kernel_height, source.kernel_width},
                "convolution weights");
        if (source.weights == nullptr || source.bias == nullptr
                || source.weight_count != expected_weights
                || source.bias_count != source.output_channels) {
            throw std::invalid_argument(
                    "optimized CPU graph segment parameter shape mismatch");
        }
        block.weights.assign(
                source.weights, source.weights + checked_size(source.weight_count, "weights"));
        block.bias.assign(
                source.bias, source.bias + checked_size(source.bias_count, "bias"));
        preceding_channels = block.output_channels;
        result.push_back(std::move(block));
    }
    return result;
}

class CompiledShape {
public:
    struct ExecutionContext;

    CompiledShape(
            const engine &cpu, const std::vector<Block> &blocks, ShapeKey input,
            memory::data_type data_type)
        : input_(input) {
        configure_threading();
        if (input_[0] <= 0 || input_[1] != blocks.front().input_channels
                || input_[2] <= 0 || input_[3] <= 0) {
            throw std::invalid_argument(
                    "optimized CPU graph input shape does not match its first block");
        }
        auto source_descriptor = memory::desc(
                memory::dims(input_.begin(), input_.end()),
                data_type, memory::format_tag::any);
        stream preparation(cpu);
        for (const auto &block : blocks) {
            auto padding_height = std::pair<std::int64_t, std::int64_t>(
                    block.padding_top, block.padding_bottom);
            auto padding_width = std::pair<std::int64_t, std::int64_t>(
                    block.padding_left, block.padding_right);
            if (block.padding_kind == 1) {
                padding_height = same_upper_padding(input_[2],
                        block.kernel_height, block.stride_height,
                        block.dilation_height);
                padding_width = same_upper_padding(input_[3],
                        block.kernel_width, block.stride_width,
                        block.dilation_width);
            }
            ShapeKey output {input_[0], block.output_channels,
                    output_dimension(input_[2], block.kernel_height,
                            block.stride_height, block.dilation_height,
                            padding_height.first, padding_height.second),
                    output_dimension(input_[3], block.kernel_width,
                            block.stride_width, block.dilation_width,
                            padding_width.first, padding_width.second)};
            const auto weight_dimensions = block.groups == 1
                    ? memory::dims {block.output_channels,
                            block.input_channels, block.kernel_height,
                            block.kernel_width}
                    : memory::dims {block.groups,
                            block.output_channels / block.groups,
                            block.input_channels / block.groups,
                            block.kernel_height, block.kernel_width};
            const auto weight_descriptor = memory::desc(weight_dimensions,
                    data_type, memory::format_tag::any);
            const auto bias_descriptor = memory::desc(
                    {block.output_channels}, memory::data_type::f32,
                    memory::format_tag::x);
            const auto destination_descriptor
                    = memory::desc(memory::dims(output.begin(), output.end()),
                            data_type, memory::format_tag::any);
            post_ops operations;
            if (block.activation_kind == 1) {
                operations.append_eltwise(
                        algorithm::eltwise_relu, 0.0f, 0.0f);
            } else if (block.activation_kind == 2) {
                operations.append_eltwise(
                        algorithm::eltwise_gelu_erf, 0.0f, 0.0f);
            }
            auto residual_post_op = -1;
            if (block.residual_source != -1) {
                residual_post_op = static_cast<int>(operations.len());
                const auto &residual_descriptor = block.residual_source == -2
                        ? descriptors_.front().src_desc()
                        : descriptors_[static_cast<std::size_t>(
                                  block.residual_source)]
                                  .dst_desc();
                operations.append_binary(
                        algorithm::binary_add, residual_descriptor);
            }
            primitive_attr attributes;
            attributes.set_scratchpad_mode(dnnl::scratchpad_mode::user);
            attributes.set_post_ops(operations);
            descriptors_.emplace_back(cpu, dnnl::prop_kind::forward_inference,
                    algorithm::convolution_direct, source_descriptor,
                    weight_descriptor, bias_descriptor, destination_descriptor,
                    memory::dims {
                            block.stride_height, block.stride_width},
                    memory::dims {block.dilation_height - 1,
                            block.dilation_width - 1},
                    memory::dims {
                            padding_height.first, padding_width.first},
                    memory::dims {
                            padding_height.second, padding_width.second},
                    attributes);
            primitives_.emplace_back(descriptors_.back());
            residual_sources_.push_back(block.residual_source);
            residual_post_ops_.push_back(residual_post_op);
            const auto user_weight_tag = block.groups == 1
                    ? memory::format_tag::oihw
                    : memory::format_tag::goihw;
            auto user_weights = memory(memory::desc(weight_dimensions,
                                               memory::data_type::f32,
                                               user_weight_tag),
                    cpu, const_cast<float *>(block.weights.data()));
            packed_weights_.emplace_back(
                    descriptors_.back().weights_desc(), cpu);
            reorder(user_weights, packed_weights_.back())
                    .execute(preparation, user_weights, packed_weights_.back());
            auto user_bias = memory(memory::desc({block.output_channels},
                                            memory::data_type::f32,
                                            memory::format_tag::x),
                    cpu, const_cast<float *>(block.bias.data()));
            biases_.emplace_back(descriptors_.back().bias_desc(), cpu);
            reorder(user_bias, biases_.back())
                    .execute(preparation, user_bias, biases_.back());
            packed_bytes_ += descriptors_.back().weights_desc().get_size();
            packed_bytes_ += descriptors_.back().bias_desc().get_size();
            scratchpad_bytes_ = std::max(scratchpad_bytes_,
                    descriptors_.back().scratchpad_desc().get_size());
            source_descriptor = descriptors_.back().dst_desc();
            input_ = output;
            outputs_.push_back(output);
        }
        preparation.wait();
        std::vector<std::size_t> activation_bytes;
        activation_bytes.reserve(descriptors_.size());
        for (const auto &descriptor : descriptors_) {
            activation_bytes.push_back(descriptor.dst_desc().get_size());
        }
        activation_plan_
                = plan_activation_arena(activation_bytes, residual_sources_);
    }

    std::size_t resident_bytes() const {
        const auto state = state_bytes();
        if (packed_bytes_ > std::numeric_limits<std::size_t>::max() - state) {
            throw std::invalid_argument(
                    "optimized CPU graph resident byte count overflowed");
        }
        return packed_bytes_ + state;
    }

    std::size_t packed_bytes() const noexcept { return packed_bytes_; }

    std::size_t output_count() const {
        return element_count(outputs_.back());
    }

    std::size_t input_count() const { return element_count(original_input()); }

    void execute(const engine &cpu, const float *input, std::size_t input_count,
            float *output, std::size_t output_count,
            std::uint64_t state_budget_bytes,
            ExecutionContext &context) const {
        configure_threading();
        if (input == nullptr || output == nullptr
                || input_count != this->input_count()
                || output_count != this->output_count()) {
            throw std::invalid_argument(
                    "optimized CPU graph execution tensor size mismatch");
        }
        if (resident_bytes() > state_budget_bytes) {
            throw std::invalid_argument(
                    "optimized CPU graph execution exceeds its state budget");
        }

        context.user_source.set_data_handle(const_cast<float *>(input));
        context.user_output.set_data_handle(output);
        context.input_reorder->execute(
                context.execution, context.user_source, context.source);
        for (std::size_t index = 0; index < descriptors_.size(); ++index) {
            const auto &block_source
                    = index == 0 ? context.source
                                 : context.activations[index - 1];
            std::unordered_map<int, memory> arguments {
                    {DNNL_ARG_SRC, block_source},
                    {DNNL_ARG_WEIGHTS, packed_weights_[index]},
                    {DNNL_ARG_BIAS, biases_[index]},
                    {DNNL_ARG_DST, context.activations[index]}};
            if (residual_sources_[index] != -1) {
                const auto &residual = residual_sources_[index] == -2
                        ? context.source
                        : context.activations[static_cast<std::size_t>(
                                  residual_sources_[index])];
                arguments.emplace(
                        DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                residual_post_ops_[index])
                                | DNNL_ARG_SRC_1,
                        residual);
            }
            if (!context.scratchpads.empty()) {
                arguments.emplace(
                        DNNL_ARG_SCRATCHPAD, context.scratchpads[index]);
            }
            primitives_[index].execute(context.execution, arguments);
        }
        context.output_reorder->execute(context.execution,
                context.activations.back(), context.user_output);
        context.execution.wait();
    }

    std::size_t state_bytes() const {
        auto result = descriptors_.front().src_desc().get_size();
        if (result
                > std::numeric_limits<std::size_t>::max()
                        - activation_plan_.total_bytes) {
            throw std::invalid_argument(
                    "optimized CPU graph execution state size overflowed");
        }
        result += activation_plan_.total_bytes;
        if (result
                > std::numeric_limits<std::size_t>::max()
                        - scratchpad_bytes_) {
            throw std::invalid_argument(
                    "optimized CPU graph scratchpad size overflowed");
        }
        return result + scratchpad_bytes_;
    }

    struct ExecutionContext {
        ExecutionContext(const engine &cpu,
                const std::vector<convolution_forward::primitive_desc>
                        &descriptors,
                const ActivationArenaPlan &activation_plan,
                const ShapeKey &input_shape, const ShapeKey &output_shape)
            : execution(cpu),
              user_source(memory::desc(
                                  memory::dims(
                                          input_shape.begin(), input_shape.end()),
                                  memory::data_type::f32,
                                  memory::format_tag::nchw),
                      cpu, DNNL_MEMORY_NONE),
              source(descriptors.front().src_desc(), cpu),
              user_output(memory::desc(
                                  memory::dims(output_shape.begin(),
                                          output_shape.end()),
                                  memory::data_type::f32,
                                  memory::format_tag::nchw),
                      cpu, DNNL_MEMORY_NONE) {
            activation_storage.reserve(
                    activation_plan.slot_capacities.size());
            for (const auto bytes : activation_plan.slot_capacities) {
                if (bytes
                        > static_cast<std::size_t>(
                                std::numeric_limits<std::int64_t>::max())) {
                    throw std::invalid_argument(
                            "optimized CPU graph activation slot is too large");
                }
                activation_storage.emplace_back(
                        memory::desc({static_cast<std::int64_t>(bytes)},
                                memory::data_type::u8, memory::format_tag::x),
                        cpu);
            }
            activations.reserve(descriptors.size());
            for (std::size_t index = 0; index < descriptors.size(); ++index) {
                const auto slot = activation_plan.activation_slots[index];
                activations.emplace_back(descriptors[index].dst_desc(), cpu,
                        activation_storage[slot].get_data_handle());
            }
            const auto scratchpad_bytes = std::accumulate(descriptors.begin(),
                    descriptors.end(), std::size_t {0},
                    [](std::size_t largest, const auto &descriptor) {
                        return std::max(largest,
                                descriptor.scratchpad_desc().get_size());
                    });
            if (scratchpad_bytes != 0) {
                if (scratchpad_bytes
                        > static_cast<std::size_t>(
                                std::numeric_limits<std::int64_t>::max())) {
                    throw std::invalid_argument(
                            "optimized CPU graph scratchpad is too large");
                }
                scratchpad_storage = std::make_unique<memory>(memory::desc(
                        {static_cast<std::int64_t>(scratchpad_bytes)},
                        memory::data_type::u8, memory::format_tag::x),
                        cpu);
                scratchpads.reserve(descriptors.size());
                for (const auto &descriptor : descriptors) {
                    scratchpads.emplace_back(descriptor.scratchpad_desc(), cpu,
                            scratchpad_storage->get_data_handle());
                }
            }
            input_reorder = std::make_unique<reorder>(user_source, source);
            output_reorder
                    = std::make_unique<reorder>(activations.back(), user_output);
        }

        stream execution;
        memory user_source;
        memory source;
        memory user_output;
        std::vector<memory> activation_storage;
        std::vector<memory> activations;
        std::unique_ptr<memory> scratchpad_storage;
        std::vector<memory> scratchpads;
        std::unique_ptr<reorder> input_reorder;
        std::unique_ptr<reorder> output_reorder;
    };

    std::unique_ptr<ExecutionContext> create_execution_context(
            const engine &cpu) const {
        return std::make_unique<ExecutionContext>(cpu, descriptors_,
                activation_plan_, original_input(), outputs_.back());
    }

private:

    ShapeKey original_input() const {
        const auto &descriptor = descriptors_.front().src_desc();
        const auto dimensions = descriptor.get_dims();
        return {dimensions[0], dimensions[1], dimensions[2], dimensions[3]};
    }

    static std::size_t element_count(const ShapeKey &shape) {
        const auto count = checked_product(
                {static_cast<std::uint64_t>(shape[0]),
                        static_cast<std::uint64_t>(shape[1]),
                        static_cast<std::uint64_t>(shape[2]),
                        static_cast<std::uint64_t>(shape[3])},
                "optimized graph tensor");
        return checked_size(count, "optimized graph tensor");
    }

    ShapeKey input_;
    std::vector<convolution_forward::primitive_desc> descriptors_;
    std::vector<primitive> primitives_;
    std::vector<memory> packed_weights_;
    std::vector<memory> biases_;
    std::vector<std::int64_t> residual_sources_;
    std::vector<int> residual_post_ops_;
    std::vector<ShapeKey> outputs_;
    ActivationArenaPlan activation_plan_;
    std::size_t packed_bytes_ = 0;
    std::size_t scratchpad_bytes_ = 0;
};

class Segment {
public:
    Segment(const NativeBlock *blocks, std::size_t block_count,
            std::uint64_t cache_budget_bytes,
            std::uint64_t context_cache_budget_bytes,
            std::uint32_t precision_kind)
        : blocks_(copy_blocks(blocks, block_count)),
          cache_budget_bytes_(checked_size(
                  cache_budget_bytes, "optimized graph cache budget")),
          context_cache_budget_bytes_(checked_size(context_cache_budget_bytes,
                  "optimized graph context cache budget")),
          data_type_(internal_data_type(precision_kind)),
          cpu_(engine::kind::cpu, 0) {}

    void execute(const ShapeKey &shape, const float *input,
            std::size_t input_count, float *output, std::size_t output_count,
            std::uint64_t state_budget_bytes) {
        auto compiled_shape = compiled(shape);
        auto slot = cached_context(shape, compiled_shape);
        if (slot) {
            std::lock_guard<std::mutex> guard(slot->execution_mutex);
            compiled_shape->execute(cpu_, input, input_count, output,
                    output_count, state_budget_bytes, *slot->context);
            return;
        }
        auto context = compiled_shape->create_execution_context(cpu_);
        compiled_shape->execute(cpu_, input, input_count, output, output_count,
                state_budget_bytes, *context);
    }

private:
    struct CacheEntry {
        std::shared_ptr<CompiledShape> compiled;
        std::size_t bytes;
        std::uint64_t last_use;
    };

    struct ExecutionSlot {
        ExecutionSlot(const engine &cpu,
                const std::shared_ptr<CompiledShape> &compiled)
            : owner(compiled),
              context(compiled->create_execution_context(cpu)) {}

        std::weak_ptr<CompiledShape> owner;
        std::unique_ptr<CompiledShape::ExecutionContext> context;
        std::mutex execution_mutex;
    };

    struct ContextEntry {
        ShapeKey shape;
        std::shared_ptr<ExecutionSlot> slot;
        std::size_t bytes;
        std::uint64_t last_use;
    };

    std::shared_ptr<CompiledShape> compiled(const ShapeKey &shape) {
        std::lock_guard<std::mutex> guard(cache_mutex_);
        const auto found = cache_.find(shape);
        if (found != cache_.end()) {
            found->second.last_use = ++cache_clock_;
            return found->second.compiled;
        }
        auto compiled = std::make_shared<CompiledShape>(
                cpu_, blocks_, shape, data_type_);
        const auto bytes = compiled->packed_bytes();
        while (!cache_.empty()
                && bytes <= cache_budget_bytes_
                && cached_bytes_ > cache_budget_bytes_ - bytes) {
            const auto oldest = std::min_element(cache_.begin(), cache_.end(),
                    [](const auto &left, const auto &right) {
                        return left.second.last_use < right.second.last_use;
                    });
            cached_bytes_ -= oldest->second.bytes;
            cache_.erase(oldest);
        }
        if (bytes <= cache_budget_bytes_) {
            cached_bytes_ += bytes;
            cache_.emplace(shape,
                    CacheEntry {
                            compiled, bytes, ++cache_clock_});
        }
        return compiled;
    }

    std::shared_ptr<ExecutionSlot> cached_context(const ShapeKey &shape,
            const std::shared_ptr<CompiledShape> &compiled) {
        std::lock_guard<std::mutex> guard(context_cache_mutex_);
        // A shape may execute concurrently. Reuse an idle context first and
        // grow a bounded pool only while its charged cache budget permits it.
        // Once the pool is full, returning a live matching slot preserves the
        // memory bound and lets the per-slot mutex provide backpressure.
        std::shared_ptr<ExecutionSlot> busy_matching_slot;
        for (auto entry = context_cache_.begin();
                entry != context_cache_.end();) {
            if (entry->shape != shape) {
                ++entry;
                continue;
            }
            const auto owner = entry->slot->owner.lock();
            if (!owner || owner.get() != compiled.get()) {
                if (entry->slot.use_count() != 1) {
                    ++entry;
                    continue;
                }
                cached_context_bytes_ -= entry->bytes;
                entry = context_cache_.erase(entry);
                continue;
            }
            if (entry->slot.use_count() == 1) {
                entry->last_use = ++context_clock_;
                return entry->slot;
            }
            if (!busy_matching_slot) busy_matching_slot = entry->slot;
            ++entry;
        }

        const auto bytes = compiled->state_bytes();
        if (bytes > context_cache_budget_bytes_) return busy_matching_slot;
        while (cached_context_bytes_ > context_cache_budget_bytes_ - bytes) {
            auto oldest = context_cache_.end();
            for (auto candidate = context_cache_.begin();
                    candidate != context_cache_.end(); ++candidate) {
                if (candidate->slot.use_count() != 1) continue;
                if (oldest == context_cache_.end()
                        || candidate->last_use < oldest->last_use) {
                    oldest = candidate;
                }
            }
            if (oldest == context_cache_.end()) return busy_matching_slot;
            cached_context_bytes_ -= oldest->bytes;
            context_cache_.erase(oldest);
        }

        auto slot = std::make_shared<ExecutionSlot>(cpu_, compiled);
        cached_context_bytes_ += bytes;
        context_cache_.push_back(
                ContextEntry {shape, slot, bytes, ++context_clock_});
        return slot;
    }

    std::vector<Block> blocks_;
    std::size_t cache_budget_bytes_;
    std::size_t context_cache_budget_bytes_;
    memory::data_type data_type_;
    engine cpu_;
    std::mutex cache_mutex_;
    std::unordered_map<ShapeKey, CacheEntry, ShapeHash> cache_;
    std::size_t cached_bytes_ = 0;
    std::uint64_t cache_clock_ = 0;
    std::mutex context_cache_mutex_;
    std::vector<ContextEntry> context_cache_;
    std::size_t cached_context_bytes_ = 0;
    std::uint64_t context_clock_ = 0;
};

void copy_error(char *destination, std::size_t capacity, const char *source) {
    if (destination == nullptr || capacity == 0) return;
    const auto length = std::min(capacity - 1, std::strlen(source));
    std::memcpy(destination, source, length);
    destination[length] = '\0';
}

template <typename Action>
std::int32_t guarded(char *error, std::size_t error_capacity, Action action) {
    if (error != nullptr && error_capacity != 0) error[0] = '\0';
    try {
        action();
        return 0;
    } catch (const dnnl::error &failure) {
        const auto detail = std::string("oneDNN status ")
                + std::to_string(static_cast<int>(failure.status)) + ": "
                + failure.what();
        copy_error(error, error_capacity, detail.c_str());
    } catch (const std::exception &failure) {
        copy_error(error, error_capacity, failure.what());
    } catch (...) {
        copy_error(error, error_capacity, "unknown native failure");
    }
    return 1;
}

} // namespace

extern "C" std::int32_t a3s_power_cpu_graph_runtime_version(
        std::int32_t *major, std::int32_t *minor, std::int32_t *patch,
        std::uint32_t *cpu_runtime) {
    const auto *version = dnnl_version();
    if (version == nullptr || major == nullptr || minor == nullptr
            || patch == nullptr || cpu_runtime == nullptr) {
        return 1;
    }
    *major = version->major;
    *minor = version->minor;
    *patch = version->patch;
    *cpu_runtime = version->cpu_runtime;
    return 0;
}

extern "C" std::int32_t a3s_power_cpu_graph_segment_create(
        const NativeBlock *blocks, std::size_t block_count,
        std::uint64_t cache_budget_bytes,
        std::uint64_t context_cache_budget_bytes, std::uint32_t precision_kind,
        void **output, char *error, std::size_t error_capacity) {
    return guarded(error, error_capacity, [&]() {
        if (output == nullptr || cache_budget_bytes == 0
                || context_cache_budget_bytes == 0) {
            throw std::invalid_argument(
                    "optimized CPU graph segment output and cache budgets are required");
        }
        *output = new Segment(blocks, block_count, cache_budget_bytes,
                context_cache_budget_bytes, precision_kind);
    });
}

extern "C" std::int32_t a3s_power_cpu_graph_segment_execute(void *segment,
        const float *input, std::size_t input_count,
        const std::uint64_t *input_dimensions, float *output,
        std::size_t output_count, std::uint64_t state_budget_bytes, char *error,
        std::size_t error_capacity) {
    return guarded(error, error_capacity, [&]() {
        if (segment == nullptr || input_dimensions == nullptr
                || state_budget_bytes == 0) {
            throw std::invalid_argument(
                    "optimized CPU graph segment execution arguments are incomplete");
        }
        const ShapeKey shape {
                checked_i64(input_dimensions[0], "batch"),
                checked_i64(input_dimensions[1], "channels"),
                checked_i64(input_dimensions[2], "height"),
                checked_i64(input_dimensions[3], "width")};
        auto *typed = static_cast<Segment *>(segment);
        typed->execute(shape, input, input_count, output, output_count,
                state_budget_bytes);
    });
}

extern "C" void a3s_power_cpu_graph_segment_destroy(void *segment) {
    delete static_cast<Segment *>(segment);
}
