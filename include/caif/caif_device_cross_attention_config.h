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
// Configuration for CAIF_DeviceCrossAttention. All five fields (dim,
// kv_input_dim, num_heads, num_kv_heads, head_dim) are required by the
// constructor so cross-attention can never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceCrossAttentionConfig:public CAIF_Base
{
  public:
    // All five fields are required.
    CAIF_DeviceCrossAttentionConfig(const uint32_t dim,
                                    const uint32_t kv_input_dim,
                                    const uint32_t num_heads,
                                    const uint32_t num_kv_heads,
                                    const uint32_t head_dim);

    // Decoder-stream width. Drives W_Q's input and W_O's output.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Encoder-output width — the width of the tensor K and V are projected
    // from. Equals `dim` for a same-width encoder-decoder pair; differs when a
    // frozen pretrained encoder of one hidden size feeds a decoder of another.
    // Drives W_K's and W_V's input.
    uint32_t KvInputDim()const{return _kv_input_dim;}
    void SetKvInputDim(const uint32_t kv_input_dim){_kv_input_dim=kv_input_dim;}

    // Number of query heads.
    uint32_t NumHeads()const{return _num_heads;}
    void SetNumHeads(const uint32_t num_heads){_num_heads=num_heads;}

    // Number of key/value heads (GQA).
    uint32_t NumKvHeads()const{return _num_kv_heads;}
    void SetNumKvHeads(const uint32_t num_kv_heads){_num_kv_heads=num_kv_heads;}

    // Per-head dimension.
    uint32_t HeadDim()const{return _head_dim;}
    void SetHeadDim(const uint32_t head_dim){_head_dim=head_dim;}

  protected:

  private:
    uint32_t _dim;
    uint32_t _kv_input_dim;
    uint32_t _num_heads;
    uint32_t _num_kv_heads;
    uint32_t _head_dim;
};

}//end instance namespace
