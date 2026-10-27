// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_kvcache_into_blocks.hpp"

#include <vector>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace {

// Information collected in the scan phase; all fields are pre-computed so the
// transform phase never needs to re-query the parameter name.
struct KVCacheTransformInfo {
    std::shared_ptr<ov::op::v0::Parameter> param;
    std::shared_ptr<ov::op::v0::Concat> concat;
    ov::Output<ov::Node> present_kv_input;
    std::shared_ptr<ov::Node> convert_node;  // nullptr when Parameter→Concat directly
    bool is_key = false;                     // true = K tensor
    bool is_value = false;                   // true = V tensor
    int64_t concat_axis = 0;                 // sequence dimension index in the concat

    // Dequantization chain of a compressed KV cache (all null for an uncompressed one):
    //   Param(i8/u8) → Convert → [Subtract(·, Convert(zp Param))] → Multiply(·, scale Param) → Concat
    std::shared_ptr<ov::op::v1::Multiply> multiply;
    std::shared_ptr<ov::op::v0::Parameter> scale_param;
    std::shared_ptr<ov::op::v1::Subtract> subtract;   // asymmetric only
    std::shared_ptr<ov::op::v0::Convert> zp_convert;  // asymmetric only
    std::shared_ptr<ov::op::v0::Parameter> zp_param;  // asymmetric only
};

// Walk a parameter's consumers to find the Concat it feeds (directly or via Convert).
// Returns {concat, convert_node}; concat is nullptr if the pattern is not found.
std::pair<std::shared_ptr<ov::op::v0::Concat>, std::shared_ptr<ov::Node>> find_concat_for_param(
    const std::shared_ptr<ov::op::v0::Parameter>& param) {
    for (const auto& output : param->outputs()) {
        for (const auto& target : output.get_target_inputs()) {
            auto node = target.get_node()->shared_from_this();
            if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(node)) {
                return {concat, nullptr};
            }
            if (ov::is_type<ov::op::v0::Convert>(node)) {
                for (const auto& cvt_out : node->outputs()) {
                    for (const auto& cvt_target : cvt_out.get_target_inputs()) {
                        auto next = cvt_target.get_node()->shared_from_this();
                        if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(next)) {
                            return {concat, node};
                        }
                    }
                }
            }
        }
    }
    return {nullptr, nullptr};
}

std::shared_ptr<ov::Node> single_user(const std::shared_ptr<ov::Node>& node) {
    const auto targets = node->output(0).get_target_inputs();
    return targets.size() == 1 ? targets.begin()->get_node()->shared_from_this() : nullptr;
}

// The DynamicQuantize scale / zero-point Parameter behind `out`, or nullptr.
std::shared_ptr<ov::op::v0::Parameter> as_dq_aux_param(const ov::Output<ov::Node>& out) {
    auto p = ov::as_type_ptr<ov::op::v0::Parameter>(out.get_node_shared_ptr());
    if (p && (ov::npuw::util::isDQScaleOrZPKey(p->get_friendly_name()) ||
              ov::npuw::util::isDQScaleOrZPValue(p->get_friendly_name()))) {
        return p;
    }
    return nullptr;
}

// The operand of a binary node that is not `chain_src`.
ov::Output<ov::Node> other_operand(const std::shared_ptr<ov::Node>& node, const std::shared_ptr<ov::Node>& chain_src) {
    return node->input_value(node->input_value(0).get_node_shared_ptr() == chain_src ? 1 : 0);
}

// Match the compressed-KV dequantization chain hanging off a data parameter:
//   Param → Convert → [Subtract(·, Convert(zp Param))] → Multiply(·, scale Param) → Concat
// On success fills the chain members of `info` and returns true.
bool find_dequant_chain(const std::shared_ptr<ov::op::v0::Parameter>& param, KVCacheTransformInfo& info) {
    auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(single_user(param));
    if (!cvt) {
        return false;
    }
    std::shared_ptr<ov::Node> chain_tail = cvt;
    std::shared_ptr<ov::Node> next = single_user(cvt);
    if (auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(next)) {
        auto zp_cvt = ov::as_type_ptr<ov::op::v0::Convert>(other_operand(sub, cvt).get_node_shared_ptr());
        auto zp = zp_cvt ? as_dq_aux_param(zp_cvt->input_value(0)) : nullptr;
        if (!zp) {
            return false;
        }
        info.subtract = sub;
        info.zp_convert = zp_cvt;
        info.zp_param = zp;
        chain_tail = sub;
        next = single_user(sub);
    }
    auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(next);
    auto scale = mul ? as_dq_aux_param(other_operand(mul, chain_tail)) : nullptr;
    auto concat = mul ? ov::as_type_ptr<ov::op::v0::Concat>(single_user(mul)) : nullptr;
    if (!scale || !concat) {
        return false;
    }
    info.convert_node = cvt;
    info.multiply = mul;
    info.scale_param = scale;
    info.concat = concat;
    return true;
}

// Trace a Concat input back to its data Parameter through Convert / Subtract / Multiply
// (the data operand is input 0 of each node of the chain).
std::shared_ptr<ov::Node> trace_to_param(std::shared_ptr<ov::Node> node) {
    while (node && (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Subtract>(node) ||
                    ov::is_type<ov::op::v1::Multiply>(node))) {
        node = node->input_value(0).get_node_shared_ptr();
    }
    return node;
}

// Build a 4D block shape.  seq_dim is the axis holding the sequence (2 or 3);
// all other dims are copied from orig_shape.
ov::Shape make_block_shape(const ov::PartialShape& orig_shape, int64_t seq_dim, size_t seq_size) {
    ov::Shape s(4);
    for (int i = 0; i < 4; ++i) {
        s[i] = (i == seq_dim) ? seq_size : static_cast<size_t>(orig_shape[i].get_length());
    }
    return s;
}

}  // namespace

SplitKVCacheIntoBlocks::SplitKVCacheIntoBlocks(uint32_t block_size, bool v_transposed)
    : m_block_size(block_size),
      m_v_transposed(v_transposed) {}

bool SplitKVCacheIntoBlocks::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool model_changed = false;

    // --- Phase 1: Scan — collect parameters that need to be split ----------
    std::vector<KVCacheTransformInfo> params_to_transform;

    for (const auto& param : model->get_parameters()) {
        const std::string& name = param->get_friendly_name();

        if (ov::npuw::util::is_swa_kv_cache_name(name)) {
            continue;
        }

        const bool is_key = ov::npuw::util::isPastKeyValuesKeyContiguous(name).has_value();
        const bool is_value = ov::npuw::util::isPastKeyValuesValueContiguous(name).has_value();
        if (!is_key && !is_value) {
            continue;  // not a contiguous KV cache parameter
        }

        // Shape must be 4D and fully static before we can proceed.
        const auto& orig_shape = param->get_partial_shape();
        if (orig_shape.rank().is_dynamic() || orig_shape.rank().get_length() != 4) {
            LOG_WARN("SplitKVCacheIntoBlocks: Skipping " << name << " — expected 4D shape, got " << orig_shape);
            continue;
        }
        if (!orig_shape[0].is_static() || !orig_shape[1].is_static() || !orig_shape[2].is_static() ||
            !orig_shape[3].is_static()) {
            LOG_WARN("SplitKVCacheIntoBlocks: Skipping " << name << " — dynamic dimensions not supported");
            continue;
        }

        // Locate the Concat this parameter feeds: either directly (optionally via Convert) or
        // through the dequantization chain of a compressed KV cache.
        KVCacheTransformInfo info;
        auto [concat, convert_node] = find_concat_for_param(param);
        if (concat) {
            info.concat = concat;
            info.convert_node = convert_node;
        } else if (!find_dequant_chain(param, info)) {
            continue;
        }

        // Locate the "present_kv" input — the non-param input of the Concat.
        bool found = false;
        for (size_t i = 0; i < info.concat->get_input_size(); ++i) {
            auto src = info.concat->input(i).get_source_output().get_node_shared_ptr();
            if (trace_to_param(src) != param) {
                info.present_kv_input = info.concat->input(i).get_source_output();
                found = true;
                break;
            }
        }
        if (!found) {
            continue;
        }

        info.param = param;
        info.is_key = is_key;
        info.is_value = is_value;
        // Pre-compute the concat axis (sequence dimension) once.
        info.concat_axis = (is_key || (is_value && !m_v_transposed)) ? 2 : 3;

        params_to_transform.push_back(std::move(info));
    }

    // --- Phase 2: Transform — replace each collected parameter with blocks --
    for (auto& info : params_to_transform) {
        auto& param = info.param;
        auto& concat = info.concat;

        const auto& orig_shape = param->get_partial_shape();
        const int64_t seq_len = orig_shape[info.concat_axis].get_length();
        const uint32_t num_full_blocks = static_cast<uint32_t>(seq_len) / m_block_size;
        const uint32_t tail_size = static_cast<uint32_t>(seq_len) % m_block_size;
        const uint32_t total_blocks = num_full_blocks + (tail_size > 0 ? 1 : 0);

        LOG_INFO("SplitKVCacheIntoBlocks: Transforming " << param->get_friendly_name() << " shape=" << orig_shape
                                                         << " → " << total_blocks << " blocks (tail=" << tail_size
                                                         << ")");

        // The scale / zero-point shapes differ from the data shape only in the embedding
        // axis, so they split along the same sequence axis at the same block boundaries.
        auto make_block_param =
            [&](const std::shared_ptr<ov::op::v0::Parameter>& src, const std::string& suffix, size_t seq_size) {
                auto block_shape = make_block_shape(src->get_partial_shape(), info.concat_axis, seq_size);
                auto p = std::make_shared<ov::op::v0::Parameter>(src->get_element_type(), block_shape);
                const std::string block_name = src->get_friendly_name() + suffix;
                p->set_friendly_name(block_name);
                p->output(0).set_names({block_name});
                return p;
            };

        // Block suffixes and sizes: full blocks followed by an optional tail.
        std::vector<std::pair<std::string, size_t>> blocks;
        blocks.reserve(total_blocks);
        for (uint32_t i = 0; i < num_full_blocks; ++i) {
            blocks.emplace_back("_block_" + std::to_string(i), m_block_size);
        }
        if (tail_size > 0) {
            blocks.emplace_back("_block_tail", tail_size);
        }

        std::vector<std::shared_ptr<ov::op::v0::Parameter>> new_params;
        ov::NodeVector new_chain_nodes;
        ov::OutputVector inputs_for_concat;
        inputs_for_concat.reserve(total_blocks + 1);

        for (const auto& [suffix, size] : blocks) {
            auto data_blk = make_block_param(param, suffix, size);
            new_params.push_back(data_blk);
            ov::Output<ov::Node> out = data_blk;

            if (info.convert_node) {
                // If the original path had a Convert, insert one after each block param.
                auto cvt = std::make_shared<ov::op::v0::Convert>(out, info.convert_node->get_output_element_type(0));
                cvt->set_friendly_name(data_blk->get_friendly_name() + "_convert");
                new_chain_nodes.push_back(cvt);
                out = cvt;
            }
            if (info.multiply) {
                // Re-create the dequantization chain per block: (x - zp) * scale.
                if (info.subtract) {
                    auto zp_blk = make_block_param(info.zp_param, suffix, size);
                    new_params.push_back(zp_blk);
                    auto zp_cvt =
                        std::make_shared<ov::op::v0::Convert>(zp_blk, info.zp_convert->get_output_element_type(0));
                    zp_cvt->set_friendly_name(zp_blk->get_friendly_name() + "_convert");
                    auto sub = std::make_shared<ov::op::v1::Subtract>(out, zp_cvt);
                    sub->set_friendly_name(info.subtract->get_friendly_name() + suffix);
                    new_chain_nodes.push_back(zp_cvt);
                    new_chain_nodes.push_back(sub);
                    out = sub;
                }
                auto scale_blk = make_block_param(info.scale_param, suffix, size);
                new_params.push_back(scale_blk);
                auto mul = std::make_shared<ov::op::v1::Multiply>(out, scale_blk);
                mul->set_friendly_name(info.multiply->get_friendly_name() + suffix);
                new_chain_nodes.push_back(mul);
                out = mul;
            }
            inputs_for_concat.push_back(out);
        }
        inputs_for_concat.push_back(info.present_kv_input);

        model->add_parameters(new_params);

        // Replace the old Concat.
        auto new_concat = std::make_shared<ov::op::v0::Concat>(inputs_for_concat, info.concat_axis);
        new_concat->set_friendly_name(concat->get_friendly_name());

        ov::NodeVector new_nodes{new_concat};
        for (const auto& p : new_params) {
            new_nodes.push_back(p);
        }
        new_nodes.insert(new_nodes.end(), new_chain_nodes.begin(), new_chain_nodes.end());
        copy_runtime_info({param, concat}, new_nodes);

        concat->output(0).replace(new_concat->output(0));
        model->remove_parameter(param);
        if (info.scale_param) {
            model->remove_parameter(info.scale_param);
        }
        if (info.zp_param) {
            model->remove_parameter(info.zp_param);
        }

        LOG_DEBUG("SplitKVCacheIntoBlocks: new concat shape " << new_concat->get_output_partial_shape(0));

        model_changed = true;
    }

    return model_changed;
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
