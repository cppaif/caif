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

//------------------------------------------------------------------------------
// CAIF - AI Framework
// Factory class that turns a runtime (compute_dtype, storage_dtype) into
// the matching CAIF_DeviceLayerNorm<ComputeT, StorageT> instantiation,
// returned as a dtype-erased std::unique_ptr<CAIF_DeviceLayer>.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_layer.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"

#include <cstdint>
#include <memory>

namespace instance
{

class CAIF_DeviceLayerNormFactory:public CAIF_Base
{
  public:
    CAIF_DeviceLayerNormFactory()=delete;

    /**
     * @brief Construct the right CAIF_DeviceLayerNorm<ComputeT, StorageT>
     * specialization for the requested dtypes. LayerNorm has no MatMul
     * (compute_dtype has no semantic effect on its kernel) but the
     * factory accepts both for cross-layer signature uniformity.
     *
     * Throws CAIF_Exception when storage_dtype isn't one of
     * Float32 / Float16 / BFloat16.
     */
    static std::unique_ptr<CAIF_DeviceLayer>
    Create(uint32_t dim,
           CAIF_CudaStream &stream,
           CAIF_DataType::CAIF_DataType_e compute_dtype,
           CAIF_DataType::CAIF_DataType_e storage_dtype,
           float epsilon);

  protected:

  private:
};

}//end instance namespace
