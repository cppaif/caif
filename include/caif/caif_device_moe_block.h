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
// AIF - AI Framework
// Device-resident MoE Transformer Block (pre-norm) layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_MOE_BLOCK_H
#define CAIF_DEVICE_MOE_BLOCK_H

#include "caif_device_layer.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_multi_head_attention.h"
#include "caif_device_moe_layer.h"
#include "caif_constants.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Pre-norm Transformer Block with MoE FFN
 *
 * Composes RMSNorm, Multi-Head Attention, and MoE Layer
 * with residual connections:
 *
 *   h   = x + attention(norm1(x))    // residual 1
 *   out = h + moe(norm2(h))          // residual 2
 *
 * This is a drop-in replacement for CAIF_DeviceTransformerBlock
 * that uses Mixture of Experts instead of a dense FFN.
 *
 * Auxiliary losses from the MoE layer (balance_loss, z_loss) are
 * accumulated and can be retrieved via LastAuxLosses().
 */
class CAIF_DeviceMoEBlock:public CAIF_DeviceLayer
{
  public:

    struct Config_t
    {
      // Attention configuration
      uint32_t dim;
      uint32_t num_heads;
      uint32_t num_kv_heads;
      float dropout_rate;
      bool causal;
      bool use_rope;
      float rope_base;

      // MoE configuration
      uint32_t num_experts;
      uint32_t top_k;
      uint32_t expert_ffn_dim;
      bool expert_use_gated;
      uint32_t num_shared_experts;
      uint32_t shared_ffn_dim;
      float capacity_factor;
      CAIF_DeviceMoELayer::OverflowStrategy_e overflow_strategy;
      float balance_loss_weight;
      float z_loss_weight;
      float router_noise_std;
    };

    struct AuxLosses_t
    {
      float balance_loss;
      float z_loss;
    };

    CAIF_DeviceMoEBlock(const Config_t &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceMoEBlock()override=default;

    // Move
    CAIF_DeviceMoEBlock(CAIF_DeviceMoEBlock &&other);
    CAIF_DeviceMoEBlock &operator=(CAIF_DeviceMoEBlock &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training)override;
    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output)override;
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // Accessors
    const Config_t &Config()const{return _config;}
    const AuxLosses_t &LastAuxLosses()const{return _last_aux_losses;}
    const CAIF_DeviceMoELayer &MoELayer()const{return *_moe;}

  protected:

  private:

    struct SubLayerMapping_t
    {
      uint32_t sub_layer_idx;
      size_t local_idx;
    };

    SubLayerMapping_t MapIndex(size_t index)const;

    Config_t _config;

    std::unique_ptr<CAIF_DeviceRMSNorm> _norm1;
    std::unique_ptr<CAIF_DeviceMultiHeadAttention> _attention;
    std::unique_ptr<CAIF_DeviceRMSNorm> _norm2;
    std::unique_ptr<CAIF_DeviceMoELayer> _moe;

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_after_attn;

    // Auxiliary losses from last forward pass
    AuxLosses_t _last_aux_losses;
};

}//end instance namespace

#endif  // CAIF_DEVICE_MOE_BLOCK_H
