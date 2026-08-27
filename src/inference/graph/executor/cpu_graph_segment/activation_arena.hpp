#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

struct ActivationArenaPlan {
    std::vector<std::size_t> activation_slots;
    std::vector<std::size_t> slot_capacities;
    std::size_t total_bytes = 0;
};

inline ActivationArenaPlan plan_activation_arena(
        const std::vector<std::size_t> &activation_bytes,
        const std::vector<std::int64_t> &residual_sources) {
    if (activation_bytes.empty()
            || activation_bytes.size() != residual_sources.size()) {
        throw std::invalid_argument(
                "optimized CPU graph activation inventory is invalid");
    }

    const auto count = activation_bytes.size();
    std::vector<std::size_t> last_use(count);
    for (std::size_t index = 0; index < count; ++index) {
        // A block output is the next block's main input. The terminal output
        // remains live until the final user-layout reorder.
        last_use[index] = index + 1 < count ? index + 1 : count;
    }
    for (std::size_t consumer = 0; consumer < count; ++consumer) {
        const auto source = residual_sources[consumer];
        if (source < 0) continue;
        const auto source_index = static_cast<std::size_t>(source);
        if (source_index >= consumer) {
            throw std::invalid_argument(
                    "optimized CPU graph residual lifetime is invalid");
        }
        last_use[source_index] = std::max(last_use[source_index], consumer);
    }

    ActivationArenaPlan plan;
    plan.activation_slots.reserve(count);
    std::vector<std::size_t> slot_last_use;
    for (std::size_t activation = 0; activation < count; ++activation) {
        const auto bytes = activation_bytes[activation];
        if (bytes == 0) {
            throw std::invalid_argument(
                    "optimized CPU graph activation cannot be empty");
        }
        auto selected = slot_last_use.size();
        auto selected_growth = std::numeric_limits<std::size_t>::max();
        auto selected_capacity = std::numeric_limits<std::size_t>::max();
        for (std::size_t slot = 0; slot < slot_last_use.size(); ++slot) {
            if (slot_last_use[slot] >= activation) continue;
            const auto capacity = plan.slot_capacities[slot];
            const auto growth = bytes > capacity ? bytes - capacity : 0;
            if (growth < selected_growth
                    || (growth == selected_growth
                            && capacity < selected_capacity)) {
                selected = slot;
                selected_growth = growth;
                selected_capacity = capacity;
            }
        }
        if (selected == slot_last_use.size()) {
            selected = slot_last_use.size();
            slot_last_use.push_back(last_use[activation]);
            plan.slot_capacities.push_back(bytes);
        } else {
            slot_last_use[selected] = last_use[activation];
            plan.slot_capacities[selected]
                    = std::max(plan.slot_capacities[selected], bytes);
        }
        plan.activation_slots.push_back(selected);
    }

    for (const auto bytes : plan.slot_capacities) {
        if (plan.total_bytes
                > std::numeric_limits<std::size_t>::max() - bytes) {
            throw std::invalid_argument(
                    "optimized CPU graph activation arena size overflowed");
        }
        plan.total_bytes += bytes;
    }
    return plan;
}
