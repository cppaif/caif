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

#include "caif_device_linear_head_factory.h"
#include "caif_device_linear_head.h"
#include "caif_exception.h"

namespace instance
{


template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeLinearHead(uint32_t input_dim,
                                                  uint32_t output_dim,
                                                  bool use_bias,
                                                  CAIF_CudaStream &stream)
{
  typename CAIF_DeviceLinearHead<ComputeT,StorageT>::Config_t cfg{input_dim,output_dim,use_bias};
  return std::make_unique<CAIF_DeviceLinearHead<ComputeT,StorageT>>(cfg,stream);
}

template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceLayer> MakeLinearHeadTied(uint32_t input_dim,
                                                      uint32_t output_dim,
                                                      bool use_bias,
                                                      CAIF_DeviceTensor &tied_weight,
                                                      CAIF_DeviceTensor &tied_weight_grad,
                                                      CAIF_CudaStream &stream)
{
  typename CAIF_DeviceLinearHead<ComputeT,StorageT>::Config_t cfg{input_dim,output_dim,use_bias};
  return std::make_unique<CAIF_DeviceLinearHead<ComputeT,StorageT>>(cfg,
                                                                     tied_weight,
                                                                     tied_weight_grad,
                                                                     stream);
}


std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceLinearHeadFactory::Create(uint32_t input_dim,
                                      uint32_t output_dim,
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
      return MakeLinearHead<float,float>(input_dim,output_dim,use_bias,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeLinearHead<float,__half>(input_dim,output_dim,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeLinearHead<float,__nv_bfloat16>(input_dim,output_dim,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeLinearHead<__half,float>(input_dim,output_dim,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeLinearHead<__half,__half>(input_dim,output_dim,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeLinearHead<__half,__nv_bfloat16>(input_dim,output_dim,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeLinearHead<__nv_bfloat16,float>(input_dim,output_dim,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeLinearHead<__nv_bfloat16,__half>(input_dim,output_dim,use_bias,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeLinearHead<__nv_bfloat16,__nv_bfloat16>(input_dim,output_dim,use_bias,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceLinearHeadFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceLinearHeadFactory::CreateTied(uint32_t input_dim,
                                          uint32_t output_dim,
                                          bool use_bias,
                                          CAIF_DeviceTensor &tied_weight,
                                          CAIF_DeviceTensor &tied_weight_grad,
                                          CAIF_CudaStream &stream,
                                          CAIF_DataType::CAIF_DataType_e compute_dtype,
                                          CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return MakeLinearHeadTied<float,float>(input_dim,output_dim,use_bias,
                                              tied_weight,tied_weight_grad,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeLinearHeadTied<float,__half>(input_dim,output_dim,use_bias,
                                               tied_weight,tied_weight_grad,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return MakeLinearHeadTied<float,__nv_bfloat16>(input_dim,output_dim,use_bias,
                                                      tied_weight,tied_weight_grad,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return MakeLinearHeadTied<__half,float>(input_dim,output_dim,use_bias,
                                               tied_weight,tied_weight_grad,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeLinearHeadTied<__half,__half>(input_dim,output_dim,use_bias,
                                                tied_weight,tied_weight_grad,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return MakeLinearHeadTied<__half,__nv_bfloat16>(input_dim,output_dim,use_bias,
                                                       tied_weight,tied_weight_grad,stream);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeLinearHeadTied<__nv_bfloat16,float>(input_dim,output_dim,use_bias,
                                                      tied_weight,tied_weight_grad,stream);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeLinearHeadTied<__nv_bfloat16,__half>(input_dim,output_dim,use_bias,
                                                       tied_weight,tied_weight_grad,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return MakeLinearHeadTied<__nv_bfloat16,__nv_bfloat16>(input_dim,output_dim,use_bias,
                                                              tied_weight,tied_weight_grad,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceLinearHeadFactory::CreateTied: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
