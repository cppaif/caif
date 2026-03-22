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
// Device-resident Transformer Model
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_TRANSFORMER_MODEL_H
#define CAIF_DEVICE_TRANSFORMER_MODEL_H

#include "caif_device_layer.h"
#include "caif_device_token_embedding.h"
#include "caif_device_positional_encoding.h"
#include "caif_device_transformer_block.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_linear_head.h"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace instance
{

/**
 * @brief Complete transformer model
 *
 * Assembles embedding, positional encoding, N transformer blocks,
 * final norm, and output head into a single trainable unit.
 *
 * Input:  [batch, seq_len] (token IDs as float)
 * Output: [batch, seq_len, output_dim]
 *
 * Components:
 *   - TokenEmbedding: [batch, seq_len] -> [batch, seq_len, dim]
 *   - PositionalEncoding (optional, not used with RoPE)
 *   - N x TransformerBlock
 *   - RMSNorm (final normalization)
 *   - LinearHead: [batch, seq_len, dim] -> [batch, seq_len, output_dim]
 */
class CAIF_DeviceTransformerModel:public CAIF_DeviceLayer
{
  public:
    /**
     * @brief Configuration for TransformerModel
     */
    struct Config_t
    {
      // Embedding
      uint32_t vocab_size;      // Vocabulary size for token embedding
      uint32_t max_seq_len;     // Maximum sequence length

      // Architecture
      uint32_t dim;             // Model dimension
      uint32_t num_heads;       // Number of attention heads
      uint32_t num_kv_heads;    // Number of KV heads (GQA), 0 = num_heads
      uint32_t num_layers;      // Number of transformer blocks
      uint32_t ffn_dim;         // FFN hidden dimension (0 = auto-compute)

      // Features
      bool causal;              // Causal attention mask
      bool use_rope;            // Use rotary position embeddings
      PositionalEncodingMode_e pe_mode;  // Learned, Sinusoidal (ignored if use_rope)

      // Output
      uint32_t output_dim;      // Output head dimension (0 = vocab_size for LM)
      bool tie_weights;         // Tie output head to embedding table
    };

    CAIF_DeviceTransformerModel(const Config_t &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceTransformerModel()override=default;

    // Move
    CAIF_DeviceTransformerModel(CAIF_DeviceTransformerModel &&other)noexcept;
    CAIF_DeviceTransformerModel &operator=(CAIF_DeviceTransformerModel &&other)noexcept;

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

  protected:

  private:
    // Map global parameter index to (component, local_index)
    void MapIndex(size_t global_index,size_t &component_idx,size_t &local_idx)const;

    Config_t _config;

    // Sub-layers
    std::unique_ptr<CAIF_DeviceTokenEmbedding> _embedding;
    std::unique_ptr<CAIF_DevicePositionalEncoding> _pos_enc;  // nullptr if RoPE
    std::vector<std::unique_ptr<CAIF_DeviceTransformerBlock>> _blocks;
    std::unique_ptr<CAIF_DeviceRMSNorm> _final_norm;
    std::unique_ptr<CAIF_DeviceLinearHead> _head;

    // Parameter counts per component (for MapIndex)
    std::vector<size_t> _param_offsets;  // Cumulative parameter counts
};

}//end instance namespace

#endif  // CAIF_DEVICE_TRANSFORMER_MODEL_H
