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
// T5-style Multi-Head Attention with relative position bias (templated on
// <ComputeT, StorageT>).
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_multi_head_attention.h"
#include "caif_device_tensor.h"
#include <cstdint>
#include <string>

namespace instance
{

template<typename ComputeT=float,typename StorageT=float>
class CAIF_DeviceT5Attention:public CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>
{
  public:
    typedef typename CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::AttentionConfig_t AttentionConfig_t;
    typedef typename CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::MHAProjections_t MHAProjections_t;

    CAIF_DeviceT5Attention(const AttentionConfig_t &config,
                           CAIF_CudaStream &stream);
    CAIF_DeviceT5Attention(const AttentionConfig_t &config,
                           MHAProjections_t projections,
                           CAIF_CudaStream &stream);
    ~CAIF_DeviceT5Attention()override=default;

    CAIF_DeviceT5Attention(CAIF_DeviceT5Attention &&other);
    CAIF_DeviceT5Attention &operator=(CAIF_DeviceT5Attention &&other);

    std::string Description()const override;

    CAIF_RunContext::Subsystem_e SubsystemTag()const override
    {
      return CAIF_RunContext::Subsystem_e::T5Attention_e;
    }

  protected:

    bool RequiresExplicitScores()const override
    {
      return true;
    }
    void ApplyScoreBias(CAIF_DeviceTensor &scores,uint32_t batch,
                        uint32_t seq_len,CAIF_RunContext &ctx)override;
    void BackwardScoreBias(const CAIF_DeviceTensor &grad_scores,
                           uint32_t batch,uint32_t seq_len,
                           CAIF_RunContext &ctx)override;

  private:
};

#ifdef USE_CAIF_CUDA
extern template class CAIF_DeviceT5Attention<float,float>;
extern template class CAIF_DeviceT5Attention<float,__half>;
extern template class CAIF_DeviceT5Attention<float,__nv_bfloat16>;
extern template class CAIF_DeviceT5Attention<__half,float>;
extern template class CAIF_DeviceT5Attention<__half,__half>;
extern template class CAIF_DeviceT5Attention<__half,__nv_bfloat16>;
extern template class CAIF_DeviceT5Attention<__nv_bfloat16,float>;
extern template class CAIF_DeviceT5Attention<__nv_bfloat16,__half>;
extern template class CAIF_DeviceT5Attention<__nv_bfloat16,__nv_bfloat16>;
#else
extern template class CAIF_DeviceT5Attention<float,float>;
#endif

}//end instance namespace
