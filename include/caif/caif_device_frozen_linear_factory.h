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
// Factory for CAIF_DeviceFrozenLinear<ComputeT, StorageT>.
//
// Selects across the 5×3 storage/compute grid:
//   StorageT ∈ {float, __half, __nv_bfloat16, int8_t, caif_int4_packed_t}
//   ComputeT ∈ {float, __half, __nv_bfloat16}
// 15 cells total. The factory throws on illegal pairs.
//
// Returns std::unique_ptr<CAIF_DeviceLayer> — caller dynamic_casts to the
// concrete CAIF_DeviceFrozenLinear<C,S> when it needs to call the
// FrozenLinear-specific methods (LoadFromTensor, LoadScalesFromHost,
// ClearFP32Cache). This matches the CAIF_DeviceDenseLayerFactory pattern.
//------------------------------------------------------------------------------
#pragma once

#include "caif_base.h"
#include "caif_device_layer.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"
#include "caif_constants.h"
#include "caif_ops.h"

#include <cstdint>
#include <memory>

namespace instance
{

class CAIF_DeviceFrozenLinearFactory:public CAIF_Base
{
  public:
    CAIF_DeviceFrozenLinearFactory()=delete;

    static std::unique_ptr<CAIF_DeviceLayer>
    Create(uint32_t input_dim,
           uint32_t output_dim,
           CAIF_DataType::CAIF_DataType_e storage_dtype,
           CAIF_CudaStream &stream,
           uint32_t group_size=g_caif_quant_default_group_size,
           bool cache_fp32=true,
           CAIF_DataType::CAIF_DataType_e compute_dtype=
             CAIF_DataType::CAIF_DataType_e::Float32,
           CAIF_Ops::QuantScheme_e int8_scheme=
             CAIF_Ops::QuantScheme_e::PerTensor_e);

  protected:

  private:
};

}//end instance namespace
