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
// CAIF_DeviceMoERouterConfig constructor. The six required fields come from the
// caller; gating_kind, norm_topk_prob and routed_scaling_factor take their
// documented defaults here in the initializer list.
//------------------------------------------------------------------------------
#include "caif_device_moe_router_config.h"

namespace instance
{

CAIF_DeviceMoERouterConfig::CAIF_DeviceMoERouterConfig(
    const uint32_t input_dim,
    const uint32_t num_experts,
    const uint32_t top_k,
    const RoutingType_e &routing_type,
    const bool use_bias,
    const float noise_std):_input_dim(input_dim),
                           _num_experts(num_experts),
                           _top_k(top_k),
                           _routing_type(routing_type),
                           _use_bias(use_bias),
                           _noise_std(noise_std),
                           _gating_kind(CAIF_DeviceMoELayerFactory::GatingKind_e::SoftmaxTopK_e),
                           _norm_topk_prob(true),
                           _routed_scaling_factor(1.0f),
                           _n_group(g_caif_moe_router_default_n_group),
                           _topk_group(g_caif_moe_router_default_topk_group),
                           _bias_update_rate(g_caif_moe_router_default_bias_update_rate)
{
}

}//end instance namespace
