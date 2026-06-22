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

#include "caif_device_ml_attention_factory.h"
#include "caif_device_ml_attention.h"
#include "caif_exception.h"

namespace instance
{


template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeMLA(uint32_t dim,
                                           uint32_t num_heads,
                                           uint32_t q_lora_rank,
                                           uint32_t kv_lora_rank,
                                           uint32_t qk_rope_head_dim,
                                           uint32_t qk_nope_head_dim,
                                           uint32_t v_head_dim,
                                           bool causal,
                                           float rope_base,
                                           int rope_style,
                                           float rms_norm_eps,
                                           CAIF_CudaStream &stream)
{
  CAIF_DeviceMLAttentionConfig cfg(dim,
                                   num_heads,
                                   q_lora_rank,
                                   kv_lora_rank,
                                   qk_rope_head_dim,
                                   qk_nope_head_dim,
                                   v_head_dim,
                                   causal,
                                   rope_base,
                                   rms_norm_eps);
  cfg.SetRopeStyle(rope_style);
  return std::make_unique<CAIF_DeviceMLAttention<ComputeT,StorageT>>(cfg,stream);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceMLAttentionFactory::Create(uint32_t dim,
                                       uint32_t num_heads,
                                       uint32_t q_lora_rank,
                                       uint32_t kv_lora_rank,
                                       uint32_t qk_rope_head_dim,
                                       uint32_t qk_nope_head_dim,
                                       uint32_t v_head_dim,
                                       bool causal,
                                       float rope_base,
                                       int rope_style,
                                       float rms_norm_eps,
                                       CAIF_CudaStream &stream,
                                       CAIF_DataType::CAIF_DataType_e compute_dtype,
                                       CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMLA<float,float>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                   qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                   causal,rope_base,rope_style,rms_norm_eps,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMLA<float,__half>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                    qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                    causal,rope_base,rope_style,rms_norm_eps,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMLA<float,__nv_bfloat16>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                           qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                           causal,rope_base,rope_style,rms_norm_eps,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMLA<__half,float>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                    qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                    causal,rope_base,rope_style,rms_norm_eps,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMLA<__half,__half>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                     qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                     causal,rope_base,rope_style,rms_norm_eps,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMLA<__half,__nv_bfloat16>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                            qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                            causal,rope_base,rope_style,rms_norm_eps,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMLA<__nv_bfloat16,float>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                           qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                           causal,rope_base,rope_style,rms_norm_eps,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMLA<__nv_bfloat16,__half>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                            qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                            causal,rope_base,rope_style,rms_norm_eps,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMLA<__nv_bfloat16,__nv_bfloat16>(dim,num_heads,q_lora_rank,kv_lora_rank,
                                                   qk_rope_head_dim,qk_nope_head_dim,v_head_dim,
                                                   causal,rope_base,rope_style,rms_norm_eps,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceMLAttentionFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
