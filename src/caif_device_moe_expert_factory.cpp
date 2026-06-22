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

#include "caif_device_moe_expert_factory.h"
#include "caif_device_moe_expert.h"
#include "caif_exception.h"

namespace instance
{


template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeExpert(uint32_t input_dim,
                                              uint32_t hidden_dim,
                                              bool use_gated,
                                              bool use_bias,
                                              CAIF_CudaStream &stream)
{
  CAIF_DeviceMoEExpertConfig cfg{input_dim,hidden_dim,use_gated,use_bias};
  return std::make_unique<CAIF_DeviceMoEExpert<ComputeT,StorageT>>(cfg,stream);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceMoEExpertFactory::Create(uint32_t input_dim,
                                     uint32_t hidden_dim,
                                     bool use_gated,
                                     bool use_bias,
                                     CAIF_CudaStream &stream,
                                     CAIF_DataType::CAIF_DataType_e compute_dtype,
                                     CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeExpert<float,float>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeExpert<float,__half>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeExpert<float,__nv_bfloat16>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeExpert<__half,float>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeExpert<__half,__half>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeExpert<__half,__nv_bfloat16>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeExpert<__nv_bfloat16,float>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeExpert<__nv_bfloat16,__half>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeExpert<__nv_bfloat16,__nv_bfloat16>(input_dim,hidden_dim,use_gated,use_bias,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceMoEExpertFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
