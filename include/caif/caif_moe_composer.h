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
// MoE convenience composer — builds MoE blocks and models from primitives
//------------------------------------------------------------------------------
#ifndef CAIF_MOE_COMPOSER_H
#define CAIF_MOE_COMPOSER_H

#include "caif_device_pre_norm_block.h"
#include "caif_device_network.h"
#include "caif_device_moe_layer.h"
#include "caif_device_positional_encoding.h"
#include "caif_cuda_stream.h"
#include "caif_data_type.h"
#include <cstdint>
#include <memory>
#include <string>

namespace instance
{

/**
 * @brief Convenience factory for MoE blocks and full MoE decoder models.
 *
 * Produces stock primitives — CAIF_DevicePreNormBlock<float,float> for blocks and
 * CAIF_DeviceNetwork for whole models — from a single configuration.
 * The composer is a shortcut, not a required abstraction layer: callers
 * that need something the configuration does not expose should bypass it
 * and hand-roll the block/model from the underlying primitives
 * (CAIF_DeviceRMSNorm, CAIF_DeviceMultiHeadAttention<float,float>, CAIF_DeviceMoELayer,
 * CAIF_DevicePreNormBlock<float,float>, CAIF_DeviceNetwork).
 *
 * Block composition matches what ANVL_TransformerBuilder assembles today:
 *
 *   CAIF_DevicePreNormBlock
 *     stage 0: (RMSNorm, CAIF_DeviceMultiHeadAttention<float,float>)
 *     stage 1: (RMSNorm, CAIF_DeviceMoELayer)
 *
 * Model composition (BuildModel):
 *
 *   TokenEmbedding
 *   [PositionalEncoding iff pe_mode != None and use_rope == false]
 *   num_layers x MoE block (via BuildMoEBlock)
 *   Final RMSNorm
 *   LinearHead (tied to embedding when tie_weights == true)
 */
class CAIF_MoEComposer
{
  public:

    struct BlockConfig_t
    {
      // Attention
      uint32_t dim;
      uint32_t num_heads;
      uint32_t num_kv_heads;
      float    attention_dropout;
      bool     causal;
      bool     use_rope;
      float    rope_base;
      int      rope_style;

      // Norms (pre-attn and pre-moe)
      float    norm_eps;

      // MoE FFN — flattened fields forwarded to CAIF_DeviceMoELayer ctor
      uint32_t moe_input_dim;
      uint32_t moe_hidden_dim;
      uint32_t moe_num_experts;
      uint32_t moe_top_k;
      bool     moe_expert_use_gated;
      bool     moe_expert_use_bias;
      uint32_t moe_num_shared_experts;
      uint32_t moe_shared_hidden_dim;
      bool     moe_router_use_bias;
      float    moe_router_noise_std;
      float    moe_capacity_factor;
      CAIF_DeviceMoELayer<float,float>::OverflowStrategy_e moe_overflow_strategy;
      float    moe_balance_loss_weight;
      float    moe_z_loss_weight;

      // Naming prefixes baked into the produced CAIF_DevicePreNormBlock<float,float>.
      // Callers supply the scheme they want (e.g. an HF-style naming
      // profile, or literal strings for tests).
      std::string norm1_prefix;
      std::string attn_prefix;
      std::string norm2_prefix;
      std::string moe_prefix;

      // Precision
      CAIF_DataType::CAIF_DataType_e storage_dtype=CAIF_DataType::CAIF_DataType_e::Float32;
      CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32;
    };

    struct ModelConfig_t
    {
      // Embedding / head
      uint32_t vocab_size;
      uint32_t max_seq_len;
      uint32_t output_dim;         // 0 = vocab_size
      bool     tie_weights;

      // Optional positional encoding (skipped when use_rope == true in block)
      PositionalEncodingMode_e pe_mode;

      // Final norm epsilon
      float    final_norm_eps;

      // Per-layer MoE block configuration — applied to every layer.
      // BuildModel uses the naming prefixes in block_template verbatim
      // for every layer; per-layer differentiation (e.g., "layers.0.*",
      // "layers.1.*") is the caller's responsibility and is applied at
      // ParameterNames(prefix) query time on each network layer.
      uint32_t      num_layers;
      BlockConfig_t block_template;

      // Precision (propagated to block_template and head at BuildModel time)
      CAIF_DataType::CAIF_DataType_e storage_dtype=CAIF_DataType::CAIF_DataType_e::Float32;
      CAIF_DataType::CAIF_DataType_e compute_dtype=CAIF_DataType::CAIF_DataType_e::Float32;
    };

    /**
     * @brief Build a single MoE transformer block.
     *
     * Equivalent to the four-line hand composition of
     * (RMSNorm, MHA, RMSNorm, MoELayer) wrapped in a CAIF_DevicePreNormBlock<float,float>.
     */
    static std::unique_ptr<CAIF_DevicePreNormBlock<float,float>>
      BuildMoEBlock(const BlockConfig_t &cfg,CAIF_CudaStream &stream);

    /**
     * @brief Build a full MoE decoder-only model.
     *
     * Every layer is an MoE block. For interleaved dense/MoE models,
     * hand-compose directly from primitives (see class comment).
     *
     * Aggregate aux loss (balance+z summed) is available via
     * CAIF_DeviceNetwork::AuxLoss(); callers that need the split should
     * bypass the composer and access the primitives directly.
     */
    static std::unique_ptr<CAIF_DeviceNetwork>
      BuildModel(const ModelConfig_t &cfg,CAIF_CudaStream &stream);

  protected:

  private:

    CAIF_MoEComposer()=delete;
};

}//end instance namespace

#endif  // CAIF_MOE_COMPOSER_H
