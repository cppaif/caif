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

#include "caif_device_frozen_linear_factory.h"
#include "caif_device_frozen_linear.h"
#include "caif_int4_packed_t.h"
#include "caif_exception.h"

namespace instance
{


typedef CAIF_DataType::CAIF_DataType_e Dtype_e;

// Build at runtime the 15-cell {StorageT × ComputeT} dispatch. Each cell
// instantiates the matching template specialization.
template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer>
MakeCell(uint32_t input_dim,
         uint32_t output_dim,
         CAIF_CudaStream &stream,
         uint32_t group_size,
         bool cache_fp32,
         CAIF_Ops::QuantScheme_e int8_scheme)
{
  return std::make_unique<CAIF_DeviceFrozenLinear<ComputeT,StorageT>>(input_dim,
                                                                       output_dim,
                                                                       stream,
                                                                       group_size,
                                                                       cache_fp32,
                                                                       int8_scheme);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceFrozenLinearFactory::Create(uint32_t input_dim,
                                       uint32_t output_dim,
                                       CAIF_DataType::CAIF_DataType_e storage_dtype,
                                       CAIF_CudaStream &stream,
                                       uint32_t group_size,
                                       bool cache_fp32,
                                       CAIF_DataType::CAIF_DataType_e compute_dtype,
                                       CAIF_Ops::QuantScheme_e int8_scheme)
{
  try
  {
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeCell<float,float>(input_dim,output_dim,stream,
                                    group_size,cache_fp32,int8_scheme);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeCell<float,__half>(input_dim,output_dim,stream,
                                     group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeCell<float,__nv_bfloat16>(input_dim,output_dim,stream,
                                            group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeCell<__half,float>(input_dim,output_dim,stream,
                                     group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeCell<__half,__half>(input_dim,output_dim,stream,
                                      group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeCell<__half,__nv_bfloat16>(input_dim,output_dim,stream,
                                             group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeCell<__nv_bfloat16,float>(input_dim,output_dim,stream,
                                            group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeCell<__nv_bfloat16,__half>(input_dim,output_dim,stream,
                                             group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeCell<__nv_bfloat16,__nv_bfloat16>(input_dim,output_dim,stream,
                                                    group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Int8&&compute_dtype==Dtype_e::Float32)
    {
      return MakeCell<float,int8_t>(input_dim,output_dim,stream,
                                     group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Int8&&compute_dtype==Dtype_e::Float16)
    {
      return MakeCell<__half,int8_t>(input_dim,output_dim,stream,
                                      group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Int8&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeCell<__nv_bfloat16,int8_t>(input_dim,output_dim,stream,
                                             group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Int4&&compute_dtype==Dtype_e::Float32)
    {
      return MakeCell<float,caif_int4_packed_t>(input_dim,output_dim,stream,
                                                 group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Int4&&compute_dtype==Dtype_e::Float16)
    {
      return MakeCell<__half,caif_int4_packed_t>(input_dim,output_dim,stream,
                                                  group_size,cache_fp32,int8_scheme);
    }
    if(storage_dtype==Dtype_e::Int4&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeCell<__nv_bfloat16,caif_int4_packed_t>(input_dim,output_dim,stream,
                                                         group_size,cache_fp32,int8_scheme);
    }
#endif
    THROW_CAIFE("CAIF_DeviceFrozenLinearFactory::Create: unsupported (storage,compute) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
