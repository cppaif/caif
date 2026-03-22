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
// Device-resident Vision Transformer (ViT) model
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_VIT_MODEL_H
#define CAIF_DEVICE_VIT_MODEL_H

#include "caif_device_layer.h"
#include "caif_device_patch_embedding.h"
#include "caif_device_positional_encoding.h"
#include "caif_device_transformer_block.h"
#include "caif_device_layernorm.h"
#include "caif_device_linear_head.h"
#include "caif_constants.h"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace instance
{

/**
 * @brief Vision Transformer (ViT) for image classification.
 *
 * Architecture:
 *   1. Patch Embedding: [B, H, W, C] -> [B, num_patches+1, dim] (with CLS token)
 *   2. Positional Encoding: add learnable position embeddings
 *   3. Transformer Blocks: N layers of self-attention + FFN
 *   4. LayerNorm: final normalization
 *   5. Classification Head: CLS token -> num_classes
 *
 * Input:  [batch, height, width, channels] (BHWC format)
 * Output: [batch, num_classes] (logits)
 *
 * Classification uses the CLS token output from position 0.
 */
class CAIF_DeviceViTModel:public CAIF_DeviceLayer
{
  public:
    struct Config_t
    {
      // Image config
      uint32_t image_height;
      uint32_t image_width;
      uint32_t channels;
      uint32_t patch_size;

      // Transformer config
      uint32_t dim;
      uint32_t num_layers;
      uint32_t num_heads;
      uint32_t ffn_hidden_dim;
      float dropout_rate;

      // Classification config
      uint32_t num_classes;

      // Optional
      bool use_rope;
      float rope_base;
    };

    CAIF_DeviceViTModel(const Config_t &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceViTModel()override=default;

    // Move
    CAIF_DeviceViTModel(CAIF_DeviceViTModel &&other);
    CAIF_DeviceViTModel &operator=(CAIF_DeviceViTModel &&other);

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
    const Config_t &GetConfig()const{return _config;}
    uint32_t NumPatches()const;
    uint32_t SequenceLength()const;  // num_patches + 1 (CLS)

    // Weight initialization
    void InitializeWeights(uint32_t seed=0);

    // Access to sublayers (for inspection/debugging)
    CAIF_DevicePatchEmbedding &PatchEmbedding(){return *_patch_embedding;}
    CAIF_DevicePositionalEncoding &PositionalEncoding(){return *_positional_encoding;}
    CAIF_DeviceTransformerBlock &TransformerBlock(size_t index);
    CAIF_DeviceLayerNorm &FinalNorm(){return *_final_norm;}
    CAIF_DeviceLinearHead &ClassificationHead(){return *_classification_head;}

  protected:

  private:
    Config_t _config;

    std::unique_ptr<CAIF_DevicePatchEmbedding> _patch_embedding;
    std::unique_ptr<CAIF_DevicePositionalEncoding> _positional_encoding;
    std::vector<std::unique_ptr<CAIF_DeviceTransformerBlock>> _transformer_blocks;
    std::unique_ptr<CAIF_DeviceLayerNorm> _final_norm;
    std::unique_ptr<CAIF_DeviceLinearHead> _classification_head;

    // Cached for backward
    CAIF_DeviceTensor _cached_cls_output;  // CLS token output before head
    uint32_t _cached_batch;
};

}//end instance namespace

#endif  // CAIF_DEVICE_VIT_MODEL_H
