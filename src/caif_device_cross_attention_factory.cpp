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

#include "caif_device_cross_attention_factory.h"
#include "caif_device_cross_attention.h"
#include "caif_exception.h"

namespace instance
{


template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeCrossAttention(uint32_t dim,
                                                      uint32_t num_heads,
                                                      uint32_t num_kv_heads,
                                                      uint32_t head_dim,
                                                      CAIF_CudaStream &stream)
{
  typename CAIF_DeviceCrossAttention<ComputeT,StorageT>::CrossAttentionConfig_t cfg{
      dim,num_heads,num_kv_heads,head_dim};
  return std::make_unique<CAIF_DeviceCrossAttention<ComputeT,StorageT>>(cfg,stream);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceCrossAttentionFactory::Create(uint32_t dim,
                                          uint32_t num_heads,
                                          uint32_t num_kv_heads,
                                          uint32_t head_dim,
                                          CAIF_CudaStream &stream,
                                          CAIF_DataType::CAIF_DataType_e compute_dtype,
                                          CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeCrossAttention<float,float>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeCrossAttention<float,__half>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeCrossAttention<float,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeCrossAttention<__half,float>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeCrossAttention<__half,__half>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeCrossAttention<__half,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeCrossAttention<__nv_bfloat16,float>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeCrossAttention<__nv_bfloat16,__half>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeCrossAttention<__nv_bfloat16,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceCrossAttentionFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
