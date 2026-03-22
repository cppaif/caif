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
// Retrainer - GLM-4.7-Flash Model Builder
// Assembles a GLM model from CAIF building blocks with optional LoRA
//------------------------------------------------------------------------------
#ifndef RTNR_GLM_MODEL_BUILDER_H
#define RTNR_GLM_MODEL_BUILDER_H

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

  /**
   * @brief GLM-4.7-Flash model configuration parsed from HuggingFace config.json
   */
  struct RTNR_GLMConfig_t
  {
    uint32_t vocab_size;
    uint32_t dim;
    uint32_t num_layers;
    uint32_t num_heads;

    // MLA dimensions
    uint32_t q_lora_rank;
    uint32_t kv_lora_rank;
    uint32_t qk_rope_head_dim;
    uint32_t qk_nope_head_dim;
    uint32_t v_head_dim;
    float rope_base;
    float rms_norm_eps;

    // Dense layer
    uint32_t dense_layer_index;
    uint32_t ffn_dim;

    // MoE config
    uint32_t moe_num_experts;
    uint32_t moe_top_k;
    uint32_t moe_hidden_dim;
    uint32_t moe_shared_experts;

    bool tie_word_embeddings;
  };

  /**
   * @brief Assembles a GLM-4.7-Flash model from CAIF building blocks.
   *
   * BuildModel and LoadWeights are instance methods that share state:
   * a map of FrozenLinear raw pointers keyed by HF weight name, populated
   * during BuildModel and consumed by LoadWeights.
   *
   * Model architecture (50 layers in CAIF_DeviceNetwork):
   *   Layer 0:     CAIF_DeviceTokenEmbedding(vocab_size, dim)
   *   Layer 1:     CAIF_DevicePreNormBlock (dense FFN, HF layer 0)
   *   Layers 2-47: CAIF_DevicePreNormBlock (MoE FFN, HF layers 1-46)
   *   Layer 48:    CAIF_DeviceRMSNorm (final norm)
   *   Layer 49:    CAIF_DeviceLinearHead (lm_head)
   *
   * Each PreNormBlock has 2 sublayers:
   *   SubLayer 0: (RMSNorm, CAIF_DeviceMLAttention with projections)
   *   SubLayer 1: (RMSNorm, CAIF_DeviceFFN or CAIF_DeviceMoELayer)
   *
   * Projections use CAIF_DeviceFrozenLinear (cache_fp32=false) for VRAM
   * efficiency, optionally wrapped with CAIF_DeviceLoRAAdapter.
   */
  class RTNR_GLMModelBuilder
  {
    public:

      /**
       * @brief Parse HuggingFace config.json into GLMConfig_t
       */
      static RTNR_GLMConfig_t ParseConfig(const std::string &config_json_path);

      /**
       * @brief Build the GLM model architecture and add layers to network.
       *        Populates internal FrozenLinear pointer map for LoadWeights.
       */
      void BuildModel(CAIF_DeviceNetwork &network,
                      CAIF_CudaStream &stream,
                      const RTNR_GLMConfig_t &config,
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
                       const RTNR_GLMConfig_t &config,
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

      std::unique_ptr<CAIF_DeviceLayer> MakeProjection(uint32_t input_dim,
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
  };

}  // namespace instance

#endif  // RTNR_GLM_MODEL_BUILDER_H
