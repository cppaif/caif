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
// Configuration for a single CAIF_MoEComposer MoE block. Every field that has
// no natural default is required by the constructor so a block can never be
// built half-configured; only the storage/compute dtypes carry documented
// defaults (set in the constructor's initializer list) and are adjusted through
// their setters.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <string>

#include "caif_base.h"
#include "caif_data_type.h"
#include "caif_device_moe_layer.h"

namespace instance
{

class CAIF_MoEComposerBlockConfig:public CAIF_Base
{
  public:
    typedef CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e OverflowStrategy_e;

    // All attention, norm, and MoE-FFN fields are required. storage_dtype and
    // compute_dtype default to Float32 and are configured via the setters.
    CAIF_MoEComposerBlockConfig(const uint32_t dim,
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
                                const std::string &moe_prefix);

    // Attention block dimension / heads.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}
    uint32_t NumHeads()const{return _num_heads;}
    void SetNumHeads(const uint32_t num_heads){_num_heads=num_heads;}
    uint32_t NumKvHeads()const{return _num_kv_heads;}
    void SetNumKvHeads(const uint32_t num_kv_heads){_num_kv_heads=num_kv_heads;}
    float AttentionDropout()const{return _attention_dropout;}
    void SetAttentionDropout(const float attention_dropout){_attention_dropout=attention_dropout;}
    bool Causal()const{return _causal;}
    void SetCausal(const bool causal){_causal=causal;}
    bool UseRope()const{return _use_rope;}
    void SetUseRope(const bool use_rope){_use_rope=use_rope;}
    float RopeBase()const{return _rope_base;}
    void SetRopeBase(const float rope_base){_rope_base=rope_base;}
    int RopeStyle()const{return _rope_style;}
    void SetRopeStyle(const int rope_style){_rope_style=rope_style;}

    // Pre-attn / pre-moe RMSNorm epsilon.
    float NormEps()const{return _norm_eps;}
    void SetNormEps(const float norm_eps){_norm_eps=norm_eps;}

    // MoE-FFN fields forwarded to the CAIF_DeviceMoELayer constructor.
    uint32_t MoeInputDim()const{return _moe_input_dim;}
    void SetMoeInputDim(const uint32_t moe_input_dim){_moe_input_dim=moe_input_dim;}
    uint32_t MoeHiddenDim()const{return _moe_hidden_dim;}
    void SetMoeHiddenDim(const uint32_t moe_hidden_dim){_moe_hidden_dim=moe_hidden_dim;}
    uint32_t MoeNumExperts()const{return _moe_num_experts;}
    void SetMoeNumExperts(const uint32_t moe_num_experts){_moe_num_experts=moe_num_experts;}
    uint32_t MoeTopK()const{return _moe_top_k;}
    void SetMoeTopK(const uint32_t moe_top_k){_moe_top_k=moe_top_k;}
    bool MoeExpertUseGated()const{return _moe_expert_use_gated;}
    void SetMoeExpertUseGated(const bool moe_expert_use_gated){_moe_expert_use_gated=moe_expert_use_gated;}
    bool MoeExpertUseBias()const{return _moe_expert_use_bias;}
    void SetMoeExpertUseBias(const bool moe_expert_use_bias){_moe_expert_use_bias=moe_expert_use_bias;}
    uint32_t MoeNumSharedExperts()const{return _moe_num_shared_experts;}
    void SetMoeNumSharedExperts(const uint32_t moe_num_shared_experts)
    {
      _moe_num_shared_experts=moe_num_shared_experts;
    }
    uint32_t MoeSharedHiddenDim()const{return _moe_shared_hidden_dim;}
    void SetMoeSharedHiddenDim(const uint32_t moe_shared_hidden_dim)
    {
      _moe_shared_hidden_dim=moe_shared_hidden_dim;
    }
    bool MoeRouterUseBias()const{return _moe_router_use_bias;}
    void SetMoeRouterUseBias(const bool moe_router_use_bias){_moe_router_use_bias=moe_router_use_bias;}
    float MoeRouterNoiseStd()const{return _moe_router_noise_std;}
    void SetMoeRouterNoiseStd(const float moe_router_noise_std){_moe_router_noise_std=moe_router_noise_std;}
    float MoeCapacityFactor()const{return _moe_capacity_factor;}
    void SetMoeCapacityFactor(const float moe_capacity_factor){_moe_capacity_factor=moe_capacity_factor;}
    OverflowStrategy_e MoeOverflowStrategy()const{return _moe_overflow_strategy;}
    void SetMoeOverflowStrategy(const OverflowStrategy_e &moe_overflow_strategy)
    {
      _moe_overflow_strategy=moe_overflow_strategy;
    }
    float MoeBalanceLossWeight()const{return _moe_balance_loss_weight;}
    void SetMoeBalanceLossWeight(const float moe_balance_loss_weight)
    {
      _moe_balance_loss_weight=moe_balance_loss_weight;
    }
    float MoeZLossWeight()const{return _moe_z_loss_weight;}
    void SetMoeZLossWeight(const float moe_z_loss_weight){_moe_z_loss_weight=moe_z_loss_weight;}

    // Naming prefixes baked into the produced CAIF_DevicePreNormBlock.
    const std::string &Norm1Prefix()const{return _norm1_prefix;}
    void SetNorm1Prefix(const std::string &norm1_prefix){_norm1_prefix=norm1_prefix;}
    const std::string &AttnPrefix()const{return _attn_prefix;}
    void SetAttnPrefix(const std::string &attn_prefix){_attn_prefix=attn_prefix;}
    const std::string &Norm2Prefix()const{return _norm2_prefix;}
    void SetNorm2Prefix(const std::string &norm2_prefix){_norm2_prefix=norm2_prefix;}
    const std::string &MoePrefix()const{return _moe_prefix;}
    void SetMoePrefix(const std::string &moe_prefix){_moe_prefix=moe_prefix;}

    // Precision (default Float32).
    CAIF_DataType::CAIF_DataType_e StorageDtype()const{return _storage_dtype;}
    void SetStorageDtype(const CAIF_DataType::CAIF_DataType_e &storage_dtype){_storage_dtype=storage_dtype;}
    CAIF_DataType::CAIF_DataType_e ComputeDtype()const{return _compute_dtype;}
    void SetComputeDtype(const CAIF_DataType::CAIF_DataType_e &compute_dtype){_compute_dtype=compute_dtype;}

  protected:

  private:
    uint32_t _dim;
    uint32_t _num_heads;
    uint32_t _num_kv_heads;
    float _attention_dropout;
    bool _causal;
    bool _use_rope;
    float _rope_base;
    int _rope_style;
    float _norm_eps;
    uint32_t _moe_input_dim;
    uint32_t _moe_hidden_dim;
    uint32_t _moe_num_experts;
    uint32_t _moe_top_k;
    bool _moe_expert_use_gated;
    bool _moe_expert_use_bias;
    uint32_t _moe_num_shared_experts;
    uint32_t _moe_shared_hidden_dim;
    bool _moe_router_use_bias;
    float _moe_router_noise_std;
    float _moe_capacity_factor;
    OverflowStrategy_e _moe_overflow_strategy;
    float _moe_balance_loss_weight;
    float _moe_z_loss_weight;
    std::string _norm1_prefix;
    std::string _attn_prefix;
    std::string _norm2_prefix;
    std::string _moe_prefix;
    CAIF_DataType::CAIF_DataType_e _storage_dtype;
    CAIF_DataType::CAIF_DataType_e _compute_dtype;
};

}//end instance namespace
