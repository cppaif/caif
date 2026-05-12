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
// Device-resident Cross-Attention layer (templated on <ComputeT, StorageT>).
//
// Q is projected from one source (decoder state); K/V from a different
// source (encoder output). No causal mask, no RoPE, no KV cache.
// 4 weight matrices: W_q, W_k, W_v, W_o.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer_typed.h"
#include "caif_device_tensor.h"
#include "caif_constants.h"
#include "caif_run_context.h"
#include "caif_data_type.h"
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceCrossAttention:public CAIF_DeviceLayerTyped<ComputeT,StorageT>
{
  public:
    struct CrossAttentionConfig_t
    {
      uint32_t dim;
      uint32_t num_heads;
      uint32_t num_kv_heads;
      uint32_t head_dim;
    };

    CAIF_DeviceCrossAttention(const CrossAttentionConfig_t &config,
                              CAIF_CudaStream &stream);
    ~CAIF_DeviceCrossAttention()override=default;

    CAIF_DeviceCrossAttention(CAIF_DeviceCrossAttention &&other);
    CAIF_DeviceCrossAttention &operator=(CAIF_DeviceCrossAttention &&other);

    CAIF_DeviceTensor ForwardCross(const CAIF_DeviceTensor &decoder_input,
                                   const CAIF_DeviceTensor &encoder_output,
                                   CAIF_RunContext &ctx);
    CAIF_DeviceTensor ForwardImpl(const CAIF_DeviceTensor &input,
                                  CAIF_RunContext &ctx)override;

    CAIF_DeviceTensor BackwardCross(const CAIF_DeviceTensor &grad_output,
                                    CAIF_DeviceTensor &grad_encoder_output,
                                    CAIF_RunContext &ctx);
    CAIF_DeviceTensor BackwardImpl(const CAIF_DeviceTensor &grad_output,
                                   CAIF_RunContext &ctx)override;

    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::CrossAttention_e;
    }

    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    const CrossAttentionConfig_t &Config()const{return _config;}
    void InitializeWeights(uint32_t seed=0)override;

  public:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype;
    using CAIF_DeviceLayer::Stream;

  protected:
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AssertInputDtype;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::AllocateOutput;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::CublasComputeType;
    using CAIF_DeviceLayerTyped<ComputeT,StorageT>::StoragePtr;

    void XavierInit(CAIF_DeviceTensor &tensor,std::mt19937 &gen,
                    uint32_t fan_in,uint32_t fan_out);

  private:
    CrossAttentionConfig_t _config;

    CAIF_DeviceTensor _w_q;
    CAIF_DeviceTensor _w_k;
    CAIF_DeviceTensor _w_v;
    CAIF_DeviceTensor _w_o;

    CAIF_DeviceTensor _grad_w_q;
    CAIF_DeviceTensor _grad_w_k;
    CAIF_DeviceTensor _grad_w_v;
    CAIF_DeviceTensor _grad_w_o;

    CAIF_DeviceTensor _cached_decoder_input;
    CAIF_DeviceTensor _cached_encoder_input;
    CAIF_DeviceTensor _cached_q_heads;
    CAIF_DeviceTensor _cached_k_heads;
    CAIF_DeviceTensor _cached_v_heads;
    CAIF_DeviceTensor _cached_concat;
    CAIF_DeviceTensor _cached_logsumexp;
    CAIF_DeviceTensor _cached_output;
    uint32_t _cached_batch;
    uint32_t _cached_dec_seq_len;
    uint32_t _cached_enc_seq_len;

    bool _use_flash_attention;
    bool _cached_use_flash;
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceCrossAttention<float,float>;
extern template class CAIF_DeviceCrossAttention<float,__half>;
extern template class CAIF_DeviceCrossAttention<float,__nv_bfloat16>;
extern template class CAIF_DeviceCrossAttention<__half,float>;
extern template class CAIF_DeviceCrossAttention<__half,__half>;
extern template class CAIF_DeviceCrossAttention<__half,__nv_bfloat16>;
extern template class CAIF_DeviceCrossAttention<__nv_bfloat16,float>;
extern template class CAIF_DeviceCrossAttention<__nv_bfloat16,__half>;
extern template class CAIF_DeviceCrossAttention<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceCrossAttention<float,float>;
#endif

}//end instance namespace
