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
// Device-resident MoE Transformer Model
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_MOE_TRANSFORMER_MODEL_H
#define CAIF_DEVICE_MOE_TRANSFORMER_MODEL_H

#include "caif_device_layer.h"
#include "caif_device_token_embedding.h"
#include "caif_device_positional_encoding.h"
#include "caif_device_transformer_block.h"
#include "caif_device_moe_block.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_linear_head.h"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace instance
{

/**
 * @brief MoE Transformer Model with interleaved dense/MoE layers
 *
 * Supports mixed architectures like Mixtral where some layers use MoE
 * and others use dense FFN. Configurable via moe_layer_interval.
 *
 * Input:  [batch, seq_len] (token IDs as float)
 * Output: [batch, seq_len, output_dim]
 *
 * Components:
 *   - TokenEmbedding: [batch, seq_len] -> [batch, seq_len, dim]
 *   - PositionalEncoding (optional, not used with RoPE)
 *   - N x (TransformerBlock or MoEBlock based on layer index)
 *   - RMSNorm (final normalization)
 *   - LinearHead: [batch, seq_len, dim] -> [batch, seq_len, output_dim]
 *
 * MoE Layer Selection:
 *   - moe_layer_interval=0: All layers are dense (no MoE)
 *   - moe_layer_interval=1: All layers use MoE
 *   - moe_layer_interval=2: Every 2nd layer uses MoE (layers 1,3,5,...)
 *   - moe_layer_interval=N: Every Nth layer uses MoE
 *
 * Auxiliary losses from all MoE blocks are accumulated and can be
 * retrieved via TotalAuxLosses() after a forward pass.
 */
class CAIF_DeviceMoETransformerModel:public CAIF_DeviceLayer
{
  public:

    struct Config_t
    {
      // Embedding
      uint32_t vocab_size;
      uint32_t max_seq_len;

      // Architecture
      uint32_t dim;
      uint32_t num_heads;
      uint32_t num_kv_heads;
      uint32_t num_layers;
      uint32_t ffn_dim;           // For dense layers (0 = auto-compute)

      // Features
      bool causal;
      bool use_rope;
      float rope_base;
      PositionalEncodingMode_e pe_mode;

      // Output
      uint32_t output_dim;
      bool tie_weights;

      // MoE configuration
      uint32_t moe_layer_interval; // 0=all dense, 1=all MoE, 2=every 2nd, etc.
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

    CAIF_DeviceMoETransformerModel(const Config_t &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceMoETransformerModel()override=default;

    // Move
    CAIF_DeviceMoETransformerModel(CAIF_DeviceMoETransformerModel &&other);
    CAIF_DeviceMoETransformerModel &operator=(CAIF_DeviceMoETransformerModel &&other);

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
    uint32_t NumLayers()const{return _config.num_layers;}
    uint32_t NumMoELayers()const{return _num_moe_layers;}
    uint32_t NumDenseLayers()const{return _config.num_layers-_num_moe_layers;}
    const AuxLosses_t &TotalAuxLosses()const{return _total_aux_losses;}

    // Check if a specific layer is MoE
    bool IsMoELayer(uint32_t layer_idx)const;

  protected:

  private:

    void MapIndex(size_t global_index,size_t &component_idx,size_t &local_idx)const;

    Config_t _config;
    uint32_t _num_moe_layers;

    // Sub-layers
    std::unique_ptr<CAIF_DeviceTokenEmbedding> _embedding;
    std::unique_ptr<CAIF_DevicePositionalEncoding> _pos_enc;

    // Mixed dense and MoE blocks - use base class pointers
    // _block_is_moe[i] indicates if _blocks[i] is MoE
    std::vector<std::unique_ptr<CAIF_DeviceLayer>> _blocks;
    std::vector<bool> _block_is_moe;

    std::unique_ptr<CAIF_DeviceRMSNorm> _final_norm;
    std::unique_ptr<CAIF_DeviceLinearHead> _head;

    // Parameter offsets for MapIndex
    std::vector<size_t> _param_offsets;

    // Accumulated auxiliary losses from last forward
    AuxLosses_t _total_aux_losses;
};

}//end instance namespace

#endif  // CAIF_DEVICE_MOE_TRANSFORMER_MODEL_H
