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
// Factory for CAIF_DeviceMultiHeadAttention<ComputeT, StorageT>.
//------------------------------------------------------------------------------
#pragma once

#include "caif_device_layer.h"
#include "caif_data_type.h"
#include "caif_cuda_stream.h"

#include <cstdint>
#include <memory>

namespace instance
{

class CAIF_DeviceMultiHeadAttentionFactory
{
  public:
    CAIF_DeviceMultiHeadAttentionFactory()=delete;
    ~CAIF_DeviceMultiHeadAttentionFactory()=delete;

    static std::unique_ptr<CAIF_DeviceLayer>
    Create(uint32_t dim,
           uint32_t num_heads,
           uint32_t num_kv_heads,
           uint32_t head_dim,
           bool causal,
           bool use_rope,
           float rope_base,
           int rope_style,
           int rope_dim,
           float dropout_rate,
           CAIF_CudaStream &stream,
           CAIF_DataType::CAIF_DataType_e compute_dtype,
           CAIF_DataType::CAIF_DataType_e storage_dtype);

    // LoRA / FrozenLinear projection variant — caller hands in pre-built Q/K/V/O
    // sublayers (typically LoRAAdapter wrappers) which the MHA owns. The four
    // projection unique_ptrs must already be the right templated cell for
    // (compute_dtype, storage_dtype); the factory does not re-cast them.
    static std::unique_ptr<CAIF_DeviceLayer>
    CreateWithProjections(uint32_t dim,
                          uint32_t num_heads,
                          uint32_t num_kv_heads,
                          uint32_t head_dim,
                          bool causal,
                          bool use_rope,
                          float rope_base,
                          int rope_style,
                          int rope_dim,
                          float dropout_rate,
                          std::unique_ptr<CAIF_DeviceLayer> q_proj,
                          std::unique_ptr<CAIF_DeviceLayer> k_proj,
                          std::unique_ptr<CAIF_DeviceLayer> v_proj,
                          std::unique_ptr<CAIF_DeviceLayer> o_proj,
                          CAIF_CudaStream &stream,
                          CAIF_DataType::CAIF_DataType_e compute_dtype,
                          CAIF_DataType::CAIF_DataType_e storage_dtype);

  protected:

  private:
};

}//end instance namespace
