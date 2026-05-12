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

#include "caif_device_dense_layer_factory.h"
#include "caif_device_dense_layer.h"
#include "caif_exception.h"

namespace instance
{

std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceDenseLayerFactory::Create(uint32_t input_size,
                                      uint32_t output_size,
                                      CAIF_DeviceActivation_e activation,
                                      CAIF_CudaStream &stream,
                                      CAIF_DataType::CAIF_DataType_e compute_dtype,
                                      CAIF_DataType::CAIF_DataType_e storage_dtype,
                                      bool use_bias)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float32)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<float,float>>(input_size,output_size,
                                                                   activation,stream,use_bias);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float32)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<float,__half>>(input_size,output_size,
                                                                    activation,stream,use_bias);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float32)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<float,__nv_bfloat16>>(input_size,output_size,
                                                                           activation,stream,use_bias);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::Float16)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<__half,float>>(input_size,output_size,
                                                                    activation,stream,use_bias);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::Float16)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<__half,__half>>(input_size,output_size,
                                                                     activation,stream,use_bias);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::Float16)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<__half,__nv_bfloat16>>(input_size,output_size,
                                                                            activation,stream,use_bias);
    }
    if(storage_dtype==Dtype_e::Float32&&compute_dtype==Dtype_e::BFloat16)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<__nv_bfloat16,float>>(input_size,output_size,
                                                                           activation,stream,use_bias);
    }
    if(storage_dtype==Dtype_e::Float16&&compute_dtype==Dtype_e::BFloat16)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<__nv_bfloat16,__half>>(input_size,output_size,
                                                                            activation,stream,use_bias);
    }
    if(storage_dtype==Dtype_e::BFloat16&&compute_dtype==Dtype_e::BFloat16)
    {
      return std::make_unique<CAIF_DeviceDenseLayer<__nv_bfloat16,__nv_bfloat16>>(input_size,output_size,
                                                                                   activation,stream,
                                                                                   use_bias);
    }
#endif
    THROW_CAIFE("CAIF_DeviceDenseLayerFactory::Create: unsupported (compute,storage) dtype pair");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
