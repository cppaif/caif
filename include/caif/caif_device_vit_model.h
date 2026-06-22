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
// CAIF_DeviceViTModel<ComputeT, StorageT> — Vision Transformer (ViT) for
// image classification.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_container.h"
#include "caif_device_vit_model_config.h"
#include "caif_device_layer_typed.h"
#include "caif_run_context.h"
#include "caif_device_patch_embedding.h"
#include "caif_device_positional_encoding.h"
#include "caif_device_transformer_block.h"
#include "caif_device_layernorm.h"
#include "caif_device_linear_head.h"
#include "caif_constants.h"
#include "caif_storage_dtype.h"
#include "caif_storage_dtype_float.h"
#ifdef USE_CAIF_CUDA
#include "caif_storage_dtype_half.h"
#include "caif_storage_dtype_bfloat16.h"
#endif

#include <cstdint>
#include <string>
#include <vector>

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
 * Classification uses the CLS token output from position 0. The CLS
 * extraction step sits between the final layernorm and the classification
 * head, so this container overrides ForwardImpl/BackwardImpl rather than
 * using the default sequential chain. Parameter iteration, zero-grad,
 * aux-loss are inherited from CAIF_DeviceContainer.
 *
 * Sublayer slot layout (fixed by ctor):
 *   [0]              patch embedding
 *   [1]              positional encoding
 *   [2 .. 1+N]       N transformer blocks
 *   [2+N]            final layernorm
 *   [3+N]            classification head
 */
template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceViTModel:public CAIF_DeviceContainer
{
  public:
    typedef CAIF_DeviceLayerTyped<ComputeT,StorageT> Typed_t;


    CAIF_DeviceViTModel(const CAIF_DeviceViTModelConfig &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceViTModel()override=default;

    // Move
    CAIF_DeviceViTModel(CAIF_DeviceViTModel &&other);
    CAIF_DeviceViTModel &operator=(CAIF_DeviceViTModel &&other);

    // CAIF_DeviceLayer interface — custom ForwardImpl/BackwardImpl (CLS split).
    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx)override;
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx)override;
    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::ViTModel_e;
    }
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // Accessors
    const CAIF_DeviceViTModelConfig &Config()const{return _config;}
    void SetConfig(const CAIF_DeviceViTModelConfig &c){_config=c;}
    const CAIF_DeviceTensor &CachedClsOutput()const{return _cached_cls_output;}
    CAIF_DeviceTensor &CachedClsOutputMutable(){return _cached_cls_output;}
    void SetCachedClsOutput(CAIF_DeviceTensor t){_cached_cls_output=std::move(t);}
    uint32_t CachedBatch()const{return _cached_batch;}
    void SetCachedBatch(const uint32_t b){_cached_batch=b;}
    uint32_t NumPatches()const;
    uint32_t SequenceLength()const;  // num_patches + 1 (CLS)

    // Weight initialization
    void InitializeWeights(uint32_t seed=0);

    // Typed accessors for inspection/debugging
    CAIF_DevicePatchEmbedding<ComputeT,StorageT> &PatchEmbedding();
    CAIF_DevicePositionalEncoding<ComputeT,StorageT> &PositionalEncoding();
    CAIF_DeviceTransformerBlock<ComputeT,StorageT> &TransformerBlock(size_t index);
    CAIF_DeviceLayerNorm<ComputeT,StorageT> &FinalNorm();
    CAIF_DeviceLinearHead<ComputeT,StorageT> &ClassificationHead();

    static constexpr CAIF_DataType::CAIF_DataType_e ComputeDtype()
    {
      return CAIF_StorageDtype_t<ComputeT>::Value;
    }
    static constexpr CAIF_DataType::CAIF_DataType_e StorageDtype()
    {
      return CAIF_StorageDtype_t<StorageT>::Value;
    }

  protected:

  private:
    CAIF_DeviceViTModelConfig _config;

    // Cached for backward — the CLS-only output fed into the classification
    // head during forward. Cached only while training so that backward can
    // route the grad through the head and back out to the full sequence.
    CAIF_DeviceTensor _cached_cls_output;
    uint32_t _cached_batch;

    // Fixed slot offsets within _sublayers.
    size_t PatchEmbeddingSlot()const{return 0;}
    size_t PositionalEncodingSlot()const{return 1;}
    size_t FirstBlockSlot()const{return 2;}
    size_t FinalNormSlot()const{return 2+Config().NumLayers();}
    size_t ClassificationHeadSlot()const{return 3+Config().NumLayers();}
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceViTModel<float,float>;
extern template class CAIF_DeviceViTModel<float,__half>;
extern template class CAIF_DeviceViTModel<float,__nv_bfloat16>;
extern template class CAIF_DeviceViTModel<__half,float>;
extern template class CAIF_DeviceViTModel<__half,__half>;
extern template class CAIF_DeviceViTModel<__half,__nv_bfloat16>;
extern template class CAIF_DeviceViTModel<__nv_bfloat16,float>;
extern template class CAIF_DeviceViTModel<__nv_bfloat16,__half>;
extern template class CAIF_DeviceViTModel<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceViTModel<float,float>;
#endif

}//end instance namespace
