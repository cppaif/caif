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

#pragma once

#include "caif_base.h"
#include "caif_device_layer.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"

#include <cstdint>
#include <memory>

namespace instance
{

class CAIF_DeviceMoELayerFactory:public CAIF_Base
{
  public:
    // Gating-math selector for the MoE router.  Default `SoftmaxTopK_e`
    // matches the historical behavior (softmax over logits, top-k
    // selection, normalised probs as combine weights). `SigmoidNoauxTc_e`
    // matches the DeepSeek-V2 / GLM-4-MoE "noaux_tc" route: top-k on
    // bias-corrected sigmoid scores, weights drawn from the original
    // sigmoid scores at the chosen indices, then optionally
    // re-normalised by `norm_topk_prob` and scaled by
    // `routed_scaling_factor`.  The runtime route() math is owned by
    // CAIF_DeviceMoERouter; this enum lives on the factory so a caller's
    // strategy/builder layer can name the regime without dragging in
    // the templated router header.
    enum class GatingKind_e:uint8_t
    {
      SoftmaxTopK_e=0,
      SigmoidNoauxTc_e=1
    };

    CAIF_DeviceMoELayerFactory()=delete;

    static std::unique_ptr<CAIF_DeviceLayer>
    Create(uint32_t input_dim,
           uint32_t hidden_dim,
           uint32_t num_experts,
           uint32_t top_k,
           bool expert_use_gated,
           bool expert_use_bias,
           uint32_t num_shared_experts,
           uint32_t shared_hidden_dim,
           bool router_use_bias,
           float router_noise_std,
           float capacity_factor,
           uint8_t overflow_strategy,
           float balance_loss_weight,
           float z_loss_weight,
           CAIF_CudaStream &stream,
           CAIF_DataType::CAIF_DataType_e compute_dtype,
           CAIF_DataType::CAIF_DataType_e storage_dtype,
           GatingKind_e gating_kind=GatingKind_e::SoftmaxTopK_e,
           bool norm_topk_prob=true,
           float routed_scaling_factor=1.0f);

  protected:

  private:
};

}//end instance namespace
