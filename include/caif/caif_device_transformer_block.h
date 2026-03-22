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
// Device-resident Transformer Block (pre-norm) layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_TRANSFORMER_BLOCK_H
#define CAIF_DEVICE_TRANSFORMER_BLOCK_H

#include "caif_device_layer.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_multi_head_attention.h"
#include "caif_device_ffn.h"
#include "caif_device_activation.h"
#include "caif_constants.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Pre-norm Transformer Block
 *
 * Composes RMSNorm, Multi-Head Attention, and FFN sub-layers
 * with residual connections:
 *
 *   h   = x + attention(norm1(x))    // residual 1
 *   out = h + ffn(norm2(h))          // residual 2
 *
 * No new CUDA kernels -- uses existing sub-layers and CAIF_DeviceOps::Add
 * for residual connections. Sub-layers cache their own inputs for backward.
 *
 * Parameters are delegated to sub-layers via MapIndex:
 *   norm1: 1 tensor (gamma)
 *   attention: 4 tensors (W_q, W_k, W_v, W_o)
 *   norm2: 1 tensor (gamma)
 *   ffn: 2 or 3 tensors (depending on gated/pointwise)
 */
class CAIF_DeviceTransformerBlock:public CAIF_DeviceLayer
{
  public:
    struct TransformerBlockConfig_t
    {
      uint32_t dim;
      uint32_t num_heads;
      uint32_t num_kv_heads;
      uint32_t ffn_dim;
      float dropout_rate;
      bool causal;
      bool use_rope;
      float rope_base;
    };

    CAIF_DeviceTransformerBlock(const TransformerBlockConfig_t &config,
                               std::unique_ptr<CAIF_DeviceActivation> activation,
                               CAIF_CudaStream &stream);

    // Convenience constructor: defaults to SwiGLU activation
    CAIF_DeviceTransformerBlock(const TransformerBlockConfig_t &config,CAIF_CudaStream &stream);

    ~CAIF_DeviceTransformerBlock()override=default;

    // Move
    CAIF_DeviceTransformerBlock(CAIF_DeviceTransformerBlock &&other);
    CAIF_DeviceTransformerBlock &operator=(CAIF_DeviceTransformerBlock &&other);

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
    const TransformerBlockConfig_t &Config()const{return _config;}
    uint32_t EffectiveFFNDim()const{return _effective_ffn_dim;}

  protected:

  private:
    struct SubLayerMapping_t
    {
      uint32_t sub_layer_idx;
      size_t local_idx;
    };

    static uint32_t ComputeDefaultFFNDim(uint32_t dim);
    SubLayerMapping_t MapIndex(size_t index)const;

    TransformerBlockConfig_t _config;
    uint32_t _effective_ffn_dim;

    std::unique_ptr<CAIF_DeviceRMSNorm> _norm1;
    std::unique_ptr<CAIF_DeviceMultiHeadAttention> _attention;
    std::unique_ptr<CAIF_DeviceRMSNorm> _norm2;
    std::unique_ptr<CAIF_DeviceFFN> _ffn;
};

}//end instance namespace

#endif  // CAIF_DEVICE_TRANSFORMER_BLOCK_H
