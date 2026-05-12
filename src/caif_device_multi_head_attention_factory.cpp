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

#include "caif_device_multi_head_attention_factory.h"
#include "caif_device_multi_head_attention.h"
#include "caif_exception.h"

namespace instance
{


template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeMHA(uint32_t dim,
                                           uint32_t num_heads,
                                           uint32_t num_kv_heads,
                                           uint32_t head_dim,
                                           bool causal,
                                           bool use_rope,
                                           float rope_base,
                                           int rope_style,
                                           int rope_dim,
                                           float dropout_rate,
                                           CAIF_CudaStream &stream)
{
  typename CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::AttentionConfig_t cfg{
      dim,num_heads,num_kv_heads,head_dim,
      causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate};
  return std::make_unique<CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>>(cfg,stream);
}

template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeMHAWithProjections(uint32_t dim,
                                                          uint32_t num_heads,
                                                          uint32_t num_kv_heads,
                                                          uint32_t head_dim,
                                                          bool causal,
                                                          bool use_rope,
                                                          float rope_base,
                                                          int rope_style,
                                                          int rope_dim,
                                                          float dropout_rate,
                                                          std::unique_ptr<CAIF_DeviceLayer> q_proj,
                                                          std::unique_ptr<CAIF_DeviceLayer> k_proj,
                                                          std::unique_ptr<CAIF_DeviceLayer> v_proj,
                                                          std::unique_ptr<CAIF_DeviceLayer> o_proj,
                                                          CAIF_CudaStream &stream)
{
  typename CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::AttentionConfig_t cfg{
      dim,num_heads,num_kv_heads,head_dim,
      causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate};
  typename CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>::MHAProjections_t proj;
  proj.q_proj=std::move(q_proj);
  proj.k_proj=std::move(k_proj);
  proj.v_proj=std::move(v_proj);
  proj.o_proj=std::move(o_proj);
  return std::make_unique<CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>>(cfg,std::move(proj),stream);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceMultiHeadAttentionFactory::Create(uint32_t dim,
                                              uint32_t num_heads,
                                              uint32_t num_kv_heads,
                                              uint32_t head_dim,
                                              bool causal,
                                              bool use_rope,
                                              float rope_base,
                                              int rope_style,
                                              int rope_dim,
                                              float dropout_rate,
                                              CAIF_CudaStream &stream,
                                              CAIF_DataType::CAIF_DataType_e compute_dtype,
                                              CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMHA<float,float>(dim,num_heads,num_kv_heads,head_dim,
                                   causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMHA<float,__half>(dim,num_heads,num_kv_heads,head_dim,
                                    causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMHA<float,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,
                                           causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMHA<__half,float>(dim,num_heads,num_kv_heads,head_dim,
                                    causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMHA<__half,__half>(dim,num_heads,num_kv_heads,head_dim,
                                     causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMHA<__half,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,
                                            causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMHA<__nv_bfloat16,float>(dim,num_heads,num_kv_heads,head_dim,
                                           causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMHA<__nv_bfloat16,__half>(dim,num_heads,num_kv_heads,head_dim,
                                            causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMHA<__nv_bfloat16,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,
                                                   causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceMultiHeadAttentionFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceMultiHeadAttentionFactory::CreateWithProjections(uint32_t dim,
                                                            uint32_t num_heads,
                                                            uint32_t num_kv_heads,
                                                            uint32_t head_dim,
                                                            bool causal,
                                                            bool use_rope,
                                                            float rope_base,
                                                            int rope_style,
                                                            int rope_dim,
                                                            float dropout_rate,
                                                            std::unique_ptr<CAIF_DeviceLayer> q_proj,
                                                            std::unique_ptr<CAIF_DeviceLayer> k_proj,
                                                            std::unique_ptr<CAIF_DeviceLayer> v_proj,
                                                            std::unique_ptr<CAIF_DeviceLayer> o_proj,
                                                            CAIF_CudaStream &stream,
                                                            CAIF_DataType::CAIF_DataType_e compute_dtype,
                                                            CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMHAWithProjections<float,float>(dim,num_heads,num_kv_heads,head_dim,
                                                  causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,
                                                  std::move(q_proj),std::move(k_proj),
                                                  std::move(v_proj),std::move(o_proj),stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMHAWithProjections<float,__half>(dim,num_heads,num_kv_heads,head_dim,
                                                   causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,
                                                   std::move(q_proj),std::move(k_proj),
                                                   std::move(v_proj),std::move(o_proj),stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMHAWithProjections<float,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,
                                                          causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,
                                                          std::move(q_proj),std::move(k_proj),
                                                          std::move(v_proj),std::move(o_proj),stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMHAWithProjections<__half,float>(dim,num_heads,num_kv_heads,head_dim,
                                                   causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,
                                                   std::move(q_proj),std::move(k_proj),
                                                   std::move(v_proj),std::move(o_proj),stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMHAWithProjections<__half,__half>(dim,num_heads,num_kv_heads,head_dim,
                                                    causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,
                                                    std::move(q_proj),std::move(k_proj),
                                                    std::move(v_proj),std::move(o_proj),stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMHAWithProjections<__half,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,
                                                           causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,
                                                           std::move(q_proj),std::move(k_proj),
                                                           std::move(v_proj),std::move(o_proj),stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMHAWithProjections<__nv_bfloat16,float>(dim,num_heads,num_kv_heads,head_dim,
                                                          causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,
                                                          std::move(q_proj),std::move(k_proj),
                                                          std::move(v_proj),std::move(o_proj),stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMHAWithProjections<__nv_bfloat16,__half>(dim,num_heads,num_kv_heads,head_dim,
                                                           causal,use_rope,rope_base,rope_style,rope_dim,dropout_rate,
                                                           std::move(q_proj),std::move(k_proj),
                                                           std::move(v_proj),std::move(o_proj),stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMHAWithProjections<__nv_bfloat16,__nv_bfloat16>(dim,num_heads,num_kv_heads,head_dim,
                                                                  causal,use_rope,rope_base,rope_style,rope_dim,
                                                                  dropout_rate,
                                                                  std::move(q_proj),std::move(k_proj),
                                                                  std::move(v_proj),std::move(o_proj),
                                                                  stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceMultiHeadAttentionFactory::CreateWithProjections: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
