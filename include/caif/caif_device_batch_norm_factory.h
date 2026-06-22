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
// the matching CAIF_DeviceBatchNorm<ComputeT, StorageT> instantiation,
// returned as a dtype-erased std::unique_ptr<CAIF_DeviceLayer>.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_layer.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"

#include <cstdint>
#include <memory>

namespace instance
{

class CAIF_DeviceBatchNormFactory:public CAIF_Base
{
  public:
    CAIF_DeviceBatchNormFactory()=delete;

    static std::unique_ptr<CAIF_DeviceLayer>
    Create(uint32_t num_features,
           CAIF_CudaStream &stream,
           CAIF_DataType::CAIF_DataType_e compute_dtype,
           CAIF_DataType::CAIF_DataType_e storage_dtype,
           float epsilon=g_caif_epsilon,
           float momentum=g_caif_default_momentum);

  protected:

  private:
};

}//end instance namespace
