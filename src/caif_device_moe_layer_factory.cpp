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

#include "caif_device_moe_layer_factory.h"
#include "caif_device_moe_layer.h"
#include "caif_exception.h"

namespace instance
{


template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeMoE(uint32_t input_dim,
                                           uint32_t hidden_dim,
                                           uint32_t num_experts,
                                           uint32_t top_k,
                                           bool expert_use_gated,
                                           bool expert_use_bias,
                                           uint32_t num_shared_experts,
                                           uint32_t shared_hidden_dim,
                                           bool router_use_bias,
                                           float router_noise_std,
                                           float capacity_factor,
                                           uint8_t overflow_strategy,
                                           float balance_loss_weight,
                                           float z_loss_weight,
                                           CAIF_CudaStream &stream,
                                           CAIF_DeviceMoELayerFactory::GatingKind_e gating_kind,
                                           bool norm_topk_prob,
                                           float routed_scaling_factor)
{
  typedef typename CAIF_DeviceMoELayer<ComputeT,StorageT>::OverflowStrategy_e OS_e;
  return std::make_unique<CAIF_DeviceMoELayer<ComputeT,StorageT>>(
      input_dim,hidden_dim,num_experts,top_k,
      expert_use_gated,expert_use_bias,
      num_shared_experts,shared_hidden_dim,
      router_use_bias,router_noise_std,
      capacity_factor,static_cast<OS_e>(overflow_strategy),
      balance_loss_weight,z_loss_weight,stream,
      gating_kind,
      norm_topk_prob,
      routed_scaling_factor);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceMoELayerFactory::Create(uint32_t input_dim,
                                    uint32_t hidden_dim,
                                    uint32_t num_experts,
                                    uint32_t top_k,
                                    bool expert_use_gated,
                                    bool expert_use_bias,
                                    uint32_t num_shared_experts,
                                    uint32_t shared_hidden_dim,
                                    bool router_use_bias,
                                    float router_noise_std,
                                    float capacity_factor,
                                    uint8_t overflow_strategy,
                                    float balance_loss_weight,
                                    float z_loss_weight,
                                    CAIF_CudaStream &stream,
                                    CAIF_DataType::CAIF_DataType_e compute_dtype,
                                    CAIF_DataType::CAIF_DataType_e storage_dtype,
                                    GatingKind_e gating_kind,
                                    bool norm_topk_prob,
                                    float routed_scaling_factor)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMoE<float,float>(input_dim,hidden_dim,num_experts,top_k,
                                   expert_use_gated,expert_use_bias,
                                   num_shared_experts,shared_hidden_dim,
                                   router_use_bias,router_noise_std,
                                   capacity_factor,overflow_strategy,
                                   balance_loss_weight,z_loss_weight,stream,
                                   gating_kind,
                                   norm_topk_prob,
                                   routed_scaling_factor);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMoE<float,__half>(input_dim,hidden_dim,num_experts,top_k,
                                    expert_use_gated,expert_use_bias,
                                    num_shared_experts,shared_hidden_dim,
                                    router_use_bias,router_noise_std,
                                    capacity_factor,overflow_strategy,
                                    balance_loss_weight,z_loss_weight,stream,
                                    gating_kind,
                                    norm_topk_prob,
                                    routed_scaling_factor);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeMoE<float,__nv_bfloat16>(input_dim,hidden_dim,num_experts,top_k,
                                           expert_use_gated,expert_use_bias,
                                           num_shared_experts,shared_hidden_dim,
                                           router_use_bias,router_noise_std,
                                           capacity_factor,overflow_strategy,
                                           balance_loss_weight,z_loss_weight,stream,
                                           gating_kind,
                                           norm_topk_prob,
                                           routed_scaling_factor);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMoE<__half,float>(input_dim,hidden_dim,num_experts,top_k,
                                    expert_use_gated,expert_use_bias,
                                    num_shared_experts,shared_hidden_dim,
                                    router_use_bias,router_noise_std,
                                    capacity_factor,overflow_strategy,
                                    balance_loss_weight,z_loss_weight,stream,
                                    gating_kind,
                                    norm_topk_prob,
                                    routed_scaling_factor);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMoE<__half,__half>(input_dim,hidden_dim,num_experts,top_k,
                                     expert_use_gated,expert_use_bias,
                                     num_shared_experts,shared_hidden_dim,
                                     router_use_bias,router_noise_std,
                                     capacity_factor,overflow_strategy,
                                     balance_loss_weight,z_loss_weight,stream,
                                     gating_kind,
                                     norm_topk_prob,
                                     routed_scaling_factor);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeMoE<__half,__nv_bfloat16>(input_dim,hidden_dim,num_experts,top_k,
                                            expert_use_gated,expert_use_bias,
                                            num_shared_experts,shared_hidden_dim,
                                            router_use_bias,router_noise_std,
                                            capacity_factor,overflow_strategy,
                                            balance_loss_weight,z_loss_weight,stream,
                                            gating_kind,
                                            norm_topk_prob,
                                            routed_scaling_factor);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMoE<__nv_bfloat16,float>(input_dim,hidden_dim,num_experts,top_k,
                                           expert_use_gated,expert_use_bias,
                                           num_shared_experts,shared_hidden_dim,
                                           router_use_bias,router_noise_std,
                                           capacity_factor,overflow_strategy,
                                           balance_loss_weight,z_loss_weight,stream,
                                           gating_kind,
                                           norm_topk_prob,
                                           routed_scaling_factor);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMoE<__nv_bfloat16,__half>(input_dim,hidden_dim,num_experts,top_k,
                                            expert_use_gated,expert_use_bias,
                                            num_shared_experts,shared_hidden_dim,
                                            router_use_bias,router_noise_std,
                                            capacity_factor,overflow_strategy,
                                            balance_loss_weight,z_loss_weight,stream,
                                            gating_kind,
                                            norm_topk_prob,
                                            routed_scaling_factor);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeMoE<__nv_bfloat16,__nv_bfloat16>(input_dim,hidden_dim,num_experts,top_k,
                                                   expert_use_gated,expert_use_bias,
                                                   num_shared_experts,shared_hidden_dim,
                                                   router_use_bias,router_noise_std,
                                                   capacity_factor,overflow_strategy,
                                                   balance_loss_weight,z_loss_weight,stream,
                                                   gating_kind,
                                                   norm_topk_prob,
                                                   routed_scaling_factor);
    }
#endif
    THROW_CAIFE("CAIF_DeviceMoELayerFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
