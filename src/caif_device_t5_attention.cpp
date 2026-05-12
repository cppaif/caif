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

#include "caif_device_t5_attention.h"
#include "caif_cuda_kernels.h"
#include "caif_exception.h"
#include <string>

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceT5Attention<ComputeT,StorageT>::CAIF_DeviceT5Attention(
                                          const AttentionConfig_t &config,
                                          CAIF_CudaStream &stream):
                                CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>(config,stream)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceT5Attention<ComputeT,StorageT>::CAIF_DeviceT5Attention(
                                          const AttentionConfig_t &config,
                                          MHAProjections_t projections,
                                          CAIF_CudaStream &stream):
                                CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>(
                                    config,std::move(projections),stream)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceT5Attention<ComputeT,StorageT>::CAIF_DeviceT5Attention(
                                          CAIF_DeviceT5Attention &&other):
                                CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>(std::move(other))
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceT5Attention<ComputeT,StorageT> &
CAIF_DeviceT5Attention<ComputeT,StorageT>::operator=(CAIF_DeviceT5Attention &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::operator=(std::move(other));
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceT5Attention<ComputeT,StorageT>::ApplyScoreBias(
                                          CAIF_DeviceTensor &scores,
                                          uint32_t batch,
                                          uint32_t seq_len,
                                          CAIF_RunContext &ctx)
{
  try
  {
    if(ctx.HasPositionBias()==false)
    {
      return;
    }
    const CAIF_DeviceTensor &position_bias=ctx.PositionBias();

    if(scores.Dtype()!=position_bias.Dtype())
    {
      THROW_CAIFE("CAIF_DeviceT5Attention::ApplyScoreBias: scores/bias dtype mismatch");
    }
    const uint32_t num_heads=this->Config().num_heads;
    const size_t head_elements=static_cast<size_t>(seq_len)*seq_len;
    const int n=static_cast<int>(head_elements);
    const cudaStream_t stream=ctx.Stream().Handle();
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t h=0;h<num_heads;++h)
      {
        const size_t so=(b*num_heads+h)*head_elements;
        const size_t bo=h*head_elements;
        launch_elementwise_add<StorageT>(scores.template DevicePtr<StorageT>()+so,
                                          position_bias.template DevicePtr<StorageT>()+bo,
                                          scores.template DevicePtr<StorageT>()+so,
                                          n,
                                          stream);
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
void CAIF_DeviceT5Attention<ComputeT,StorageT>::BackwardScoreBias(
                                          const CAIF_DeviceTensor &grad_scores,
                                          uint32_t batch,
                                          uint32_t seq_len,
                                          CAIF_RunContext &ctx)
{
  try
  {
    if(ctx.HasGradPositionBias()==false || ctx.HasPositionBias()==false)
    {
      return;
    }
    CAIF_DeviceTensor &grad_position_bias=ctx.GradPositionBias();

    if(grad_scores.Dtype()!=grad_position_bias.Dtype())
    {
      THROW_CAIFE("CAIF_DeviceT5Attention::BackwardScoreBias: grad_scores/grad_bias dtype mismatch");
    }
    const uint32_t num_heads=this->Config().num_heads;
    const size_t head_elements=static_cast<size_t>(seq_len)*seq_len;
    const int n=static_cast<int>(head_elements);
    const cudaStream_t stream=ctx.Stream().Handle();
    for(uint32_t b=0;b<batch;++b)
    {
      for(uint32_t h=0;h<num_heads;++h)
      {
        const size_t so=(b*num_heads+h)*head_elements;
        const size_t bo=h*head_elements;
        launch_elementwise_add<StorageT>(grad_position_bias.template DevicePtr<StorageT>()+bo,
                                          grad_scores.template DevicePtr<StorageT>()+so,
                                          grad_position_bias.template DevicePtr<StorageT>()+bo,
                                          n,
                                          stream);
      }
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceT5Attention<ComputeT,StorageT>::Description()const
{
  try
  {
    return "T5Attention("+CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::Description()+")";
  }
  CAIF_CATCH_BLOCK()
}

template class CAIF_DeviceT5Attention<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceT5Attention<float,__half>;
template class CAIF_DeviceT5Attention<float,__nv_bfloat16>;
template class CAIF_DeviceT5Attention<__half,float>;
template class CAIF_DeviceT5Attention<__half,__half>;
template class CAIF_DeviceT5Attention<__half,__nv_bfloat16>;
template class CAIF_DeviceT5Attention<__nv_bfloat16,float>;
template class CAIF_DeviceT5Attention<__nv_bfloat16,__half>;
template class CAIF_DeviceT5Attention<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
