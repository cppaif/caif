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
// Configuration for CAIF_DeviceMLAttention (Multi-head Latent Attention). The
// ten architecture-defining fields are required by the constructor so MLA can
// never be built half-configured. rope_style carries a documented default (set
// in the constructor's initializer list, not as an in-class initializer) and is
// adjusted through its setter.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceMLAttentionConfig:public CAIF_Base
{
  public:
    // The ten required, architecture-defining fields. rope_style (RoPE layout)
    // takes its documented default and is set via SetRopeStyle().
    CAIF_DeviceMLAttentionConfig(const uint32_t dim,
                                 const uint32_t num_heads,
                                 const uint32_t q_lora_rank,
                                 const uint32_t kv_lora_rank,
                                 const uint32_t qk_rope_head_dim,
                                 const uint32_t qk_nope_head_dim,
                                 const uint32_t v_head_dim,
                                 const bool causal,
                                 const float rope_base,
                                 const float rms_norm_eps);

    // Model (hidden) dimension.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Number of attention heads.
    uint32_t NumHeads()const{return _num_heads;}
    void SetNumHeads(const uint32_t num_heads){_num_heads=num_heads;}

    // Query low-rank compression dimension.
    uint32_t QLoraRank()const{return _q_lora_rank;}
    void SetQLoraRank(const uint32_t q_lora_rank){_q_lora_rank=q_lora_rank;}

    // Key/value low-rank compression dimension.
    uint32_t KvLoraRank()const{return _kv_lora_rank;}
    void SetKvLoraRank(const uint32_t kv_lora_rank){_kv_lora_rank=kv_lora_rank;}

    // Per-head dimension of the RoPE-carrying part of Q/K.
    uint32_t QkRopeHeadDim()const{return _qk_rope_head_dim;}
    void SetQkRopeHeadDim(const uint32_t qk_rope_head_dim){_qk_rope_head_dim=qk_rope_head_dim;}

    // Per-head dimension of the non-RoPE (NoPE) part of Q/K.
    uint32_t QkNopeHeadDim()const{return _qk_nope_head_dim;}
    void SetQkNopeHeadDim(const uint32_t qk_nope_head_dim){_qk_nope_head_dim=qk_nope_head_dim;}

    // Per-head dimension of V.
    uint32_t VHeadDim()const{return _v_head_dim;}
    void SetVHeadDim(const uint32_t v_head_dim){_v_head_dim=v_head_dim;}

    // Causal (autoregressive) attention mask.
    bool Causal()const{return _causal;}
    void SetCausal(const bool causal){_causal=causal;}

    // RoPE base frequency (theta).
    float RopeBase()const{return _rope_base;}
    void SetRopeBase(const float rope_base){_rope_base=rope_base;}

    // RMSNorm epsilon for the Q/KV-compression norms.
    float RmsNormEps()const{return _rms_norm_eps;}
    void SetRmsNormEps(const float rms_norm_eps){_rms_norm_eps=rms_norm_eps;}

    // RoPE layout: 0 = interleaved, 1 = half-split. Defaults to 0.
    int RopeStyle()const{return _rope_style;}
    void SetRopeStyle(const int rope_style){_rope_style=rope_style;}

    // Cached-decode dispatch: once the KV cache reaches this length, eligible
    // single-token decode switches from the per-step decompress path to the
    // matrix-absorption path (attention in the normed-latent space). Below it
    // the decompress path wins; above it absorption's context-flat cost wins.
    // The ctor computes this from the model shape (~ dim/(qk_nope+v_head) times
    // the GPU's GEMM:bandwidth ratio); override per deployment with the setter.
    uint32_t DecodeAbsorbThreshold()const{return _decode_absorb_threshold;}
    void SetDecodeAbsorbThreshold(const uint32_t v){_decode_absorb_threshold=v;}

  protected:

  private:
    uint32_t _dim;
    uint32_t _num_heads;
    uint32_t _q_lora_rank;
    uint32_t _kv_lora_rank;
    uint32_t _qk_rope_head_dim;
    uint32_t _qk_nope_head_dim;
    uint32_t _v_head_dim;
    bool _causal;
    float _rope_base;
    float _rms_norm_eps;
    int _rope_style;
    uint32_t _decode_absorb_threshold;
};

}//end instance namespace
