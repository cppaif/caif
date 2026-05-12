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

#include "caif_device_moe_router_factory.h"
#include "caif_device_moe_router.h"
#include "caif_exception.h"

namespace instance
{


template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeRouter(uint32_t input_dim,
                                              uint32_t num_experts,
                                              uint32_t top_k,
                                              uint8_t routing_type,
                                              bool use_bias,
                                              float noise_std,
                                              CAIF_CudaStream &stream)
{
  typename CAIF_DeviceMoERouter<ComputeT,StorageT>::Config_t cfg{
      input_dim,num_experts,top_k,
      static_cast<typename CAIF_DeviceMoERouter<ComputeT,StorageT>::RoutingType_e>(routing_type),
      use_bias,noise_std};
  return std::make_unique<CAIF_DeviceMoERouter<ComputeT,StorageT>>(cfg,stream);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceMoERouterFactory::Create(uint32_t input_dim,
                                     uint32_t num_experts,
                                     uint32_t top_k,
                                     uint8_t routing_type,
                                     bool use_bias,
                                     float noise_std,
                                     CAIF_CudaStream &stream,
                                     CAIF_DataType::CAIF_DataType_e compute_dtype,
                                     CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeRouter<float,float>(input_dim,num_experts,top_k,routing_type,
                                      use_bias,noise_std,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeRouter<float,__half>(input_dim,num_experts,top_k,routing_type,
                                       use_bias,noise_std,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeRouter<float,__nv_bfloat16>(input_dim,num_experts,top_k,routing_type,
                                              use_bias,noise_std,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeRouter<__half,float>(input_dim,num_experts,top_k,routing_type,
                                       use_bias,noise_std,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeRouter<__half,__half>(input_dim,num_experts,top_k,routing_type,
                                        use_bias,noise_std,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeRouter<__half,__nv_bfloat16>(input_dim,num_experts,top_k,routing_type,
                                               use_bias,noise_std,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeRouter<__nv_bfloat16,float>(input_dim,num_experts,top_k,routing_type,
                                              use_bias,noise_std,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeRouter<__nv_bfloat16,__half>(input_dim,num_experts,top_k,routing_type,
                                               use_bias,noise_std,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeRouter<__nv_bfloat16,__nv_bfloat16>(input_dim,num_experts,top_k,routing_type,
                                                      use_bias,noise_std,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceMoERouterFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
