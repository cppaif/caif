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

#include "caif_device_average_pooling2d_factory.h"
#include "caif_device_average_pooling2d.h"
#include "caif_exception.h"

namespace instance
{

std::unique_ptr<CAIF_DeviceLayer>
CAIF_DeviceAveragePooling2DFactory::Create(uint32_t pool_height,
                                            uint32_t pool_width,
                                            uint32_t stride_height,
                                            uint32_t stride_width,
                                            CAIF_CudaStream &stream,
                                            CAIF_DataType::CAIF_DataType_e compute_dtype,
                                            CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    typedef CAIF_DataType::CAIF_DataType_e Dtype_e;
    (void)compute_dtype;
    if(storage_dtype==Dtype_e::Float32)
    {
      CAIF_DeviceAveragePooling2D<float,float>::Config_t cfg{pool_height,pool_width,
                                                              stride_height,stride_width};
      return std::make_unique<CAIF_DeviceAveragePooling2D<float,float>>(cfg,stream);
    }
#ifdef USE_CAIF_CUDA
    if(storage_dtype==Dtype_e::Float16)
    {
      CAIF_DeviceAveragePooling2D<__half,__half>::Config_t cfg{pool_height,pool_width,
                                                                stride_height,stride_width};
      return std::make_unique<CAIF_DeviceAveragePooling2D<__half,__half>>(cfg,stream);
    }
    if(storage_dtype==Dtype_e::BFloat16)
    {
      CAIF_DeviceAveragePooling2D<__nv_bfloat16,__nv_bfloat16>::Config_t
                                            cfg{pool_height,pool_width,stride_height,stride_width};
      return std::make_unique<CAIF_DeviceAveragePooling2D<__nv_bfloat16,
                                                            __nv_bfloat16>>(cfg,stream);
    }
#endif
    THROW_CAIFE("CAIF_DeviceAveragePooling2DFactory::Create: unsupported storage_dtype");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
