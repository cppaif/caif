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

#include "caif_device_layernorm_factory.h"
#include "caif_device_layernorm.h"
#include "caif_exception.h"

namespace instance
{

std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceLayerNormFactory::Create(uint32_t dim,
                                    CAIF_CudaStream &stream,
                                    CAIF_DataType::CAIF_DataType_e compute_dtype,
                                    CAIF_DataType::CAIF_DataType_e storage_dtype,
                                    float epsilon)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    (void)compute_dtype;
    if(storage_dtype==Dtype_e::Float32)
    {
      return std::make_unique<CAIF_DeviceLayerNorm<float,float>>(dim,stream,epsilon);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16)
    {
      return std::make_unique<CAIF_DeviceLayerNorm<__half,__half>>(dim,stream,epsilon);
    }
    if(storage_dtype==Dtype_e::BFloat16)
    {
      return std::make_unique<CAIF_DeviceLayerNorm<__nv_bfloat16,__nv_bfloat16>>(dim,stream,epsilon);
    }
#endif
    THROW_CAIFE("CAIF_DeviceLayerNormFactory::Create: unsupported storage_dtype");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
