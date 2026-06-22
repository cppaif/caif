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
// Factory for CAIF_DeviceCrossAttention<ComputeT, StorageT>.
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

class CAIF_DeviceCrossAttentionFactory:public CAIF_Base
{
  public:
    CAIF_DeviceCrossAttentionFactory()=delete;

    // Same-width encoder/decoder pair — the encoder output is `dim` wide,
    // so K/V project from `dim` just like Q. Delegates to the explicit
    // overload below with kv_input_dim == dim.
    static std::unique_ptr<CAIF_DeviceLayer>
    Create(uint32_t dim,
           uint32_t num_heads,
           uint32_t num_kv_heads,
           uint32_t head_dim,
           CAIF_CudaStream &stream,
           CAIF_DataType::CAIF_DataType_e compute_dtype,
           CAIF_DataType::CAIF_DataType_e storage_dtype);

    // Different-width encoder/decoder pair — `kv_input_dim` is the
    // encoder-output width that W_K / W_V project from; `dim` stays the
    // decoder-stream width (W_Q / W_O). Use this when a frozen pretrained
    // encoder of one hidden size feeds a decoder of another.
    static std::unique_ptr<CAIF_DeviceLayer>
    Create(uint32_t dim,
           uint32_t kv_input_dim,
           uint32_t num_heads,
           uint32_t num_kv_heads,
           uint32_t head_dim,
           CAIF_CudaStream &stream,
           CAIF_DataType::CAIF_DataType_e compute_dtype,
           CAIF_DataType::CAIF_DataType_e storage_dtype);

  protected:

  private:
    // Per-(compute,storage)-dtype constructor. Private because the dtype
    // dispatch belongs to Create; grouped on the factory class rather
    // than left as a free helper in the .cpp.
    template<typename ComputeT,typename StorageT>
    static std::unique_ptr<CAIF_DeviceLayer>
    MakeCrossAttention(uint32_t dim,
                       uint32_t kv_input_dim,
                       uint32_t num_heads,
                       uint32_t num_kv_heads,
                       uint32_t head_dim,
                       CAIF_CudaStream &stream);
};

}//end instance namespace
