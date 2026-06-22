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
// Factory for CAIF_DeviceMLAttention<ComputeT, StorageT>.
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

class CAIF_DeviceMLAttentionFactory:public CAIF_Base
{
  public:
    CAIF_DeviceMLAttentionFactory()=delete;

    static std::unique_ptr<CAIF_DeviceLayer>
    Create(uint32_t dim,
           uint32_t num_heads,
           uint32_t q_lora_rank,
           uint32_t kv_lora_rank,
           uint32_t qk_rope_head_dim,
           uint32_t qk_nope_head_dim,
           uint32_t v_head_dim,
           bool causal,
           float rope_base,
           int rope_style,
           float rms_norm_eps,
           CAIF_CudaStream &stream,
           CAIF_DataType::CAIF_DataType_e compute_dtype,
           CAIF_DataType::CAIF_DataType_e storage_dtype);

  protected:

  private:
};

}//end instance namespace
