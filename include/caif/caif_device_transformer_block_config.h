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
// Configuration for CAIF_DeviceTransformerBlock. The eight architecture-
// defining fields are required by the constructor so a block can never be built
// half-configured. rope_style carries a documented default (set in the
// constructor's initializer list, not as an in-class initializer) and is
// adjusted through its setter.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceTransformerBlockConfig:public CAIF_Base
{
  public:
    // The eight required, architecture-defining fields. rope_style (RoPE
    // layout) takes its documented default and is set via SetRopeStyle().
    CAIF_DeviceTransformerBlockConfig(const uint32_t dim,
                                      const uint32_t num_heads,
                                      const uint32_t num_kv_heads,
                                      const uint32_t ffn_dim,
                                      const float dropout_rate,
                                      const bool causal,
                                      const bool use_rope,
                                      const float rope_base);

    // Model (hidden) dimension.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Number of query heads.
    uint32_t NumHeads()const{return _num_heads;}
    void SetNumHeads(const uint32_t num_heads){_num_heads=num_heads;}

    // Number of key/value heads (GQA).
    uint32_t NumKvHeads()const{return _num_kv_heads;}
    void SetNumKvHeads(const uint32_t num_kv_heads){_num_kv_heads=num_kv_heads;}

    // FFN hidden dimension (0 = auto-compute from dim).
    uint32_t FfnDim()const{return _ffn_dim;}
    void SetFfnDim(const uint32_t ffn_dim){_ffn_dim=ffn_dim;}

    // Attention/FFN dropout rate.
    float DropoutRate()const{return _dropout_rate;}
    void SetDropoutRate(const float dropout_rate){_dropout_rate=dropout_rate;}

    // Causal (autoregressive) attention mask.
    bool Causal()const{return _causal;}
    void SetCausal(const bool causal){_causal=causal;}

    // Apply rotary position embeddings.
    bool UseRope()const{return _use_rope;}
    void SetUseRope(const bool use_rope){_use_rope=use_rope;}

    // RoPE base frequency (theta).
    float RopeBase()const{return _rope_base;}
    void SetRopeBase(const float rope_base){_rope_base=rope_base;}

    // RoPE layout: 0 = interleaved, 1 = half-split. Defaults to 0.
    int RopeStyle()const{return _rope_style;}
    void SetRopeStyle(const int rope_style){_rope_style=rope_style;}

  protected:

  private:
    uint32_t _dim;
    uint32_t _num_heads;
    uint32_t _num_kv_heads;
    uint32_t _ffn_dim;
    float _dropout_rate;
    bool _causal;
    bool _use_rope;
    float _rope_base;
    int _rope_style;
};

}//end instance namespace
