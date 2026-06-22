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
// Configuration for CAIF_DeviceMoERouter. The six architecture-defining fields
// (input_dim, num_experts, top_k, routing_type, use_bias, noise_std) are
// required by the constructor so a router can never be built half-configured.
// The gating regime, top-k renormalisation flag, and routed-scaling factor
// carry documented defaults (set in the constructor's initializer list, not as
// in-class initializers) and are adjusted through their setters.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"
#include "caif_device_moe_layer_factory.h"

namespace instance
{

// Documented defaults for the setter-configured fields, set in the ctor's
// initializer list. n_group 1 = no group routing; topk_group 0 = all groups
// (only meaningful when n_group > 1); bias_update_rate 0 = no online
// aux-loss-free bias update (the loaded bias stays static).
constexpr uint32_t g_caif_moe_router_default_n_group=1;
constexpr uint32_t g_caif_moe_router_default_topk_group=0;
constexpr float g_caif_moe_router_default_bias_update_rate=0.0f;

class CAIF_DeviceMoERouterConfig:public CAIF_Base
{
  public:
    // Routing algorithm used to assign tokens to experts.
    enum class RoutingType_e:uint8_t
    {
      TopK=0,
      ExpertChoice=1,
      Soft=2
    };

    // The six required, architecture-defining fields. The optional gating_kind,
    // norm_topk_prob and routed_scaling_factor fields take their documented
    // defaults and are configured via the setters below.
    CAIF_DeviceMoERouterConfig(const uint32_t input_dim,
                               const uint32_t num_experts,
                               const uint32_t top_k,
                               const RoutingType_e &routing_type,
                               const bool use_bias,
                               const float noise_std);

    // Router input dimension (model hidden size).
    uint32_t InputDim()const{return _input_dim;}
    void SetInputDim(const uint32_t input_dim){_input_dim=input_dim;}

    // Total number of experts to route across.
    uint32_t NumExperts()const{return _num_experts;}
    void SetNumExperts(const uint32_t num_experts){_num_experts=num_experts;}

    // Number of experts each token is routed to.
    uint32_t TopK()const{return _top_k;}
    void SetTopK(const uint32_t top_k){_top_k=top_k;}

    // Routing algorithm.
    RoutingType_e RoutingType()const{return _routing_type;}
    void SetRoutingType(const RoutingType_e &routing_type){_routing_type=routing_type;}

    // Add a learnable router bias term.
    bool UseBias()const{return _use_bias;}
    void SetUseBias(const bool use_bias){_use_bias=use_bias;}

    // Standard deviation of the training-time router noise (0 = none).
    float NoiseStd()const{return _noise_std;}
    void SetNoiseStd(const float noise_std){_noise_std=noise_std;}

    // Gating regime — see CAIF_DeviceMoELayerFactory::GatingKind_e for the
    // semantics. Defaults to SoftmaxTopK_e.
    CAIF_DeviceMoELayerFactory::GatingKind_e GatingKind()const{return _gating_kind;}
    void SetGatingKind(const CAIF_DeviceMoELayerFactory::GatingKind_e &gating_kind){_gating_kind=gating_kind;}

    // Re-normalise the selected top-k weights to sum=1 after gating. True
    // matches HF DeepSeek-V2 / GLM-4-MoE behavior. Defaults to true.
    bool NormTopkProb()const{return _norm_topk_prob;}
    void SetNormTopkProb(const bool norm_topk_prob){_norm_topk_prob=norm_topk_prob;}

    // Post-normalise multiplicative scale on the combine weights. 1.0 matches
    // the un-scaled HF default; DeepSeek-V3 / GLM variants use values > 1.
    float RoutedScalingFactor()const{return _routed_scaling_factor;}
    void SetRoutedScalingFactor(const float routed_scaling_factor){_routed_scaling_factor=routed_scaling_factor;}

    // DeepSeek group-limited routing: split the experts into n_group equal
    // groups, score each group by the sum of its top-2 expert scores, keep only
    // the top-topk_group groups, and mask the rest before the top-k expert
    // select. n_group = 1 (default) disables grouping. topk_group must be in
    // [1, n_group] when grouping is on. num_experts must be divisible by n_group.
    uint32_t NGroup()const{return _n_group;}
    void SetNGroup(const uint32_t n_group){_n_group=n_group;}

    uint32_t TopkGroup()const{return _topk_group;}
    void SetTopkGroup(const uint32_t topk_group){_topk_group=topk_group;}

    // Aux-loss-free load balancing (DeepSeek-V3): per training step, each
    // expert's router bias is nudged by +/- this rate from its observed token
    // load (no auxiliary loss, no gradient). 0 (default) leaves the loaded bias
    // static. Requires use_bias and SigmoidNoauxTc gating.
    float BiasUpdateRate()const{return _bias_update_rate;}
    void SetBiasUpdateRate(const float bias_update_rate){_bias_update_rate=bias_update_rate;}

  protected:

  private:
    uint32_t _input_dim;
    uint32_t _num_experts;
    uint32_t _top_k;
    RoutingType_e _routing_type;
    bool _use_bias;
    float _noise_std;
    CAIF_DeviceMoELayerFactory::GatingKind_e _gating_kind;
    bool _norm_topk_prob;
    float _routed_scaling_factor;
    uint32_t _n_group;
    uint32_t _topk_group;
    float _bias_update_rate;
};

}//end instance namespace
