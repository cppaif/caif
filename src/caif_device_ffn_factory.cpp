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

#include "caif_device_ffn_factory.h"
#include "caif_device_ffn.h"
#include "caif_exception.h"

namespace instance
{


template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeFFN(uint32_t dim,
                                           uint32_t ffn_dim,
                                           std::unique_ptr<CAIF_DeviceActivation> activation,
                                           CAIF_CudaStream &stream)
{
  typename CAIF_DeviceFFN<ComputeT,StorageT>::FFNConfig_t cfg{dim,ffn_dim};
  return std::make_unique<CAIF_DeviceFFN<ComputeT,StorageT>>(cfg,std::move(activation),stream);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceFFNFactory::Create(uint32_t dim,
                               uint32_t ffn_dim,
                               std::unique_ptr<CAIF_DeviceActivation> activation,
                               CAIF_CudaStream &stream,
                               CAIF_DataType::CAIF_DataType_e compute_dtype,
                               CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeFFN<float,float>(dim,ffn_dim,std::move(activation),stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeFFN<float,__half>(dim,ffn_dim,std::move(activation),stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeFFN<float,__nv_bfloat16>(dim,ffn_dim,std::move(activation),stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeFFN<__half,float>(dim,ffn_dim,std::move(activation),stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeFFN<__half,__half>(dim,ffn_dim,std::move(activation),stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeFFN<__half,__nv_bfloat16>(dim,ffn_dim,std::move(activation),stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeFFN<__nv_bfloat16,float>(dim,ffn_dim,std::move(activation),stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeFFN<__nv_bfloat16,__half>(dim,ffn_dim,std::move(activation),stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeFFN<__nv_bfloat16,__nv_bfloat16>(dim,ffn_dim,std::move(activation),stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceFFNFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
