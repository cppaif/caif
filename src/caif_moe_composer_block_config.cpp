// Copyright 2026 Eric Malloy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//------------------------------------------------------------------------------
// CAIF - AI Framework
// CAIF_MoEComposerBlockConfig constructor. Every attention / norm / MoE-FFN
// field is required and comes from the caller; storage_dtype and compute_dtype
// take their Float32 defaults here in the initializer list.
//------------------------------------------------------------------------------
#include "caif_moe_composer_block_config.h"

namespace instance
{

CAIF_MoEComposerBlockConfig::CAIF_MoEComposerBlockConfig(
    const uint32_t dim,
    const uint32_t num_heads,
    const uint32_t num_kv_heads,
    const float attention_dropout,
    const bool causal,
    const bool use_rope,
    const float rope_base,
    const int rope_style,
    const float norm_eps,
    const uint32_t moe_input_dim,
    const uint32_t moe_hidden_dim,
    const uint32_t moe_num_experts,
    const uint32_t moe_top_k,
    const bool moe_expert_use_gated,
    const bool moe_expert_use_bias,
    const uint32_t moe_num_shared_experts,
    const uint32_t moe_shared_hidden_dim,
    const bool moe_router_use_bias,
    const float moe_router_noise_std,
    const float moe_capacity_factor,
    const OverflowStrategy_e &moe_overflow_strategy,
    const float moe_balance_loss_weight,
    const float moe_z_loss_weight,
    const std::string &norm1_prefix,
    const std::string &attn_prefix,
    const std::string &norm2_prefix,
    const std::string &moe_prefix):_dim(dim),
                                   _num_heads(num_heads),
                                   _num_kv_heads(num_kv_heads),
                                   _attention_dropout(attention_dropout),
                                   _causal(causal),
                                   _use_rope(use_rope),
                                   _rope_base(rope_base),
                                   _rope_style(rope_style),
                                   _norm_eps(norm_eps),
                                   _moe_input_dim(moe_input_dim),
                                   _moe_hidden_dim(moe_hidden_dim),
                                   _moe_num_experts(moe_num_experts),
                                   _moe_top_k(moe_top_k),
                                   _moe_expert_use_gated(moe_expert_use_gated),
                                   _moe_expert_use_bias(moe_expert_use_bias),
                                   _moe_num_shared_experts(moe_num_shared_experts),
                                   _moe_shared_hidden_dim(moe_shared_hidden_dim),
                                   _moe_router_use_bias(moe_router_use_bias),
                                   _moe_router_noise_std(moe_router_noise_std),
                                   _moe_capacity_factor(moe_capacity_factor),
                                   _moe_overflow_strategy(moe_overflow_strategy),
                                   _moe_balance_loss_weight(moe_balance_loss_weight),
                                   _moe_z_loss_weight(moe_z_loss_weight),
                                   _norm1_prefix(norm1_prefix),
                                   _attn_prefix(attn_prefix),
                                   _norm2_prefix(norm2_prefix),
                                   _moe_prefix(moe_prefix),
                                   _storage_dtype(CAIF_DataType::CAIF_DataType_e::Float32),
                                   _compute_dtype(CAIF_DataType::CAIF_DataType_e::Float32)
{
}

}//end instance namespace
