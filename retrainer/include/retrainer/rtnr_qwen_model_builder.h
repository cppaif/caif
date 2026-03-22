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
// Retrainer - Qwen2.5-Coder-1.5B Model Builder
// Assembles a Qwen model from CAIF building blocks with optional LoRA
//------------------------------------------------------------------------------
#ifndef RTNR_QWEN_MODEL_BUILDER_H
#define RTNR_QWEN_MODEL_BUILDER_H

#include "rtnr_exception.h"

#include "caif/caif_device_network.h"
#include "caif/caif_device_tensor.h"
#include "caif/caif_data_type.h"
#include "caif/caif_cuda_stream.h"

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace instance
{

  class CAIF_DeviceFrozenLinear;
  class CAIF_DeviceMultiHeadAttention;

  /**
   * @brief Qwen2.5-Coder-1.5B model configuration parsed from HuggingFace config.json
   */
  struct RTNR_QwenConfig_t
  {
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t ffn_dim;
    float rope_base;
    float rms_norm_eps;
    bool tie_word_embeddings;
    bool use_qkv_bias;
  };

  /**
   * @brief Assembles a Qwen2.5-Coder-1.5B model from CAIF building blocks.
   *
   * BuildModel and LoadWeights are instance methods that share state:
   * a map of FrozenLinear raw pointers keyed by HF weight name, populated
   * during BuildModel and consumed by LoadWeights.
   *
   * Model architecture (num_layers+3 layers in CAIF_DeviceNetwork):
   *   Layer 0:                  CAIF_DeviceTokenEmbedding(vocab_size, dim)
   *   Layers 1..num_layers:     CAIF_DevicePreNormBlock (dense SwiGLU FFN)
   *   Layer num_layers+1:       CAIF_DeviceRMSNorm (final norm)
   *   Layer num_layers+2:       CAIF_DeviceLinearHead (lm_head, tied weights)
   *
   * Each PreNormBlock has 2 sublayers:
   *   SubLayer 0: (RMSNorm, CAIF_DeviceGQAttention with Q/K/V bias, no O bias)
   *   SubLayer 1: (RMSNorm, CAIF_DeviceFFN with SwiGLU activation)
   *
   * Projections use CAIF_DeviceFrozenLinear (cache_fp32=false) for VRAM
   * efficiency, optionally wrapped with CAIF_DeviceLoRAAdapter.
   *
   * Qwen uses standard GQA (grouped-query attention) instead of MLA.
   * tie_word_embeddings=true shares embedding weights with lm_head.
   */
  class RTNR_QwenModelBuilder
  {
    public:

      /**
       * @brief Parse HuggingFace config.json into RTNR_QwenConfig_t
       */
      static RTNR_QwenConfig_t ParseConfig(const std::string &config_json_path);

      /**
       * @brief Build the Qwen model architecture and add layers to network.
       *        Populates internal FrozenLinear pointer map for LoadWeights.
       */
      void BuildModel(CAIF_DeviceNetwork &network,
                      CAIF_CudaStream &stream,
                      const RTNR_QwenConfig_t &config,
                      CAIF_DataType::CAIF_DataType_e storage_dtype,
                      uint32_t lora_rank=0,
                      float lora_alpha=0.0f,
                      const std::vector<std::string> &lora_targets={});

      /**
       * @brief Load weights from HuggingFace sharded safetensors directory.
       *        Uses FrozenLinear pointer map populated by BuildModel.
       */
      void LoadWeights(CAIF_DeviceNetwork &network,
                       CAIF_CudaStream &stream,
                       const std::string &model_dir,
                       const RTNR_QwenConfig_t &config,
                       CAIF_DataType::CAIF_DataType_e storage_dtype);

      /**
       * @brief Save only LoRA weights to a safetensors file
       */
      static void SaveLoRAWeights(const CAIF_DeviceNetwork &network,
                                  const std::string &path);

      /**
       * @brief Load LoRA weights from a safetensors file
       */
      static void LoadLoRAWeights(CAIF_DeviceNetwork &network,
                                  const std::string &path,
                                  CAIF_CudaStream &stream);

    protected:

    private:

      static bool IsLoRATarget(const std::string &name,
                               const std::vector<std::string> &targets);

      std::unique_ptr<CAIF_DeviceLayer> MakeProjection(
        uint32_t input_dim,
        uint32_t output_dim,
        CAIF_DataType::CAIF_DataType_e dtype,
        CAIF_CudaStream &stream,
        const std::string &proj_name,
        const std::string &hf_weight_name,
        uint32_t lora_rank,
        float lora_alpha,
        const std::vector<std::string> &lora_targets);

      // Map of HF weight name -> FrozenLinear raw pointer, populated by BuildModel
      std::map<std::string,CAIF_DeviceFrozenLinear*> _weight_map;

      // MHA layer raw pointers for bias loading in LoadWeights
      std::vector<CAIF_DeviceMultiHeadAttention*> _mha_layers;
  };

}  // namespace instance

#endif  // RTNR_QWEN_MODEL_BUILDER_H
