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
// Configuration for CAIF_DeviceMultiHeadAttention. The eight architecture-
// defining fields (dim, num_heads, num_kv_heads, head_dim, causal, use_rope,
// rope_base, dropout_rate) are required by the constructor so attention can
// never be built half-configured. rope_style, rope_dim and qk_norm_eps carry
// documented defaults (set in the constructor's initializer list, not as
// in-class initializers) and are adjusted through their setters.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

// Documented defaults for the setter-configured fields, set in the ctor's
// initializer list. RoPE layout 0 = interleaved; rope_dim 0 = full rotation;
// QK-norm epsilon = standard RMSNorm eps; attention logit soft-cap 0 = disabled.
constexpr int g_caif_mha_default_rope_style=0;
constexpr int g_caif_mha_default_rope_dim=0;
constexpr float g_caif_mha_default_qk_norm_eps=1.0e-5f;
constexpr float g_caif_mha_default_attn_logit_softcap=0.0f;
constexpr int g_caif_mha_default_sliding_window=0;
constexpr bool g_caif_mha_default_use_alibi=false;

class CAIF_DeviceMultiHeadAttentionConfig:public CAIF_Base
{
  public:
    // The eight required, architecture-defining fields. rope_style (RoPE
    // layout), rope_dim (partial-rotary width) and qk_norm_eps (QK-norm
    // epsilon) take their documented defaults and are set via the setters.
    CAIF_DeviceMultiHeadAttentionConfig(const uint32_t dim,
                                        const uint32_t num_heads,
                                        const uint32_t num_kv_heads,
                                        const uint32_t head_dim,
                                        const bool causal,
                                        const bool use_rope,
                                        const float rope_base,
                                        const float dropout_rate);

    // Model (hidden) dimension.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Number of query heads.
    uint32_t NumHeads()const{return _num_heads;}
    void SetNumHeads(const uint32_t num_heads){_num_heads=num_heads;}

    // Number of key/value heads (GQA; equal to num_heads for standard MHA).
    uint32_t NumKvHeads()const{return _num_kv_heads;}
    void SetNumKvHeads(const uint32_t num_kv_heads){_num_kv_heads=num_kv_heads;}

    // Per-head dimension.
    uint32_t HeadDim()const{return _head_dim;}
    void SetHeadDim(const uint32_t head_dim){_head_dim=head_dim;}

    // Causal (autoregressive) attention mask.
    bool Causal()const{return _causal;}
    void SetCausal(const bool causal){_causal=causal;}

    // Apply rotary position embeddings to Q and K.
    bool UseRope()const{return _use_rope;}
    void SetUseRope(const bool use_rope){_use_rope=use_rope;}

    // RoPE base frequency (theta).
    float RopeBase()const{return _rope_base;}
    void SetRopeBase(const float rope_base){_rope_base=rope_base;}

    // Attention dropout rate.
    float DropoutRate()const{return _dropout_rate;}
    void SetDropoutRate(const float dropout_rate){_dropout_rate=dropout_rate;}

    // RoPE layout: 0 = interleaved, 1 = half-split. Defaults to 0.
    int RopeStyle()const{return _rope_style;}
    void SetRopeStyle(const int rope_style){_rope_style=rope_style;}

    // Number of leading head dims to rotate (0 = full rotation). Used for
    // partial-rotary models like Glm4Moe. Defaults to 0.
    int RopeDim()const{return _rope_dim;}
    void SetRopeDim(const int rope_dim){_rope_dim=rope_dim;}

    // RMSNorm epsilon used when QK-norm gammas are loaded (OLMoE / Qwen3 /
    // Olmo2). Defaults to 1e-5. No effect when no QK-norm gammas are present.
    float QkNormEps()const{return _qk_norm_eps;}
    void SetQkNormEps(const float qk_norm_eps){_qk_norm_eps=qk_norm_eps;}

    // Attention logit soft-cap (Gemma-2/3): scores = cap*tanh(scores/cap),
    // applied after the 1/sqrt(head_dim) scale and before the mask/softmax.
    // Defaults to 0.0, which disables it (no soft-cap).
    float AttnLogitSoftcap()const{return _attn_logit_softcap;}
    void SetAttnLogitSoftcap(const float attn_logit_softcap){_attn_logit_softcap=attn_logit_softcap;}

    // Sliding-window attention (Mistral, Gemma-2 alternating layers): query q
    // attends only to keys in [q-window+1, q]. Defaults to 0, which disables it
    // (full causal/non-causal attention).
    int SlidingWindow()const{return _sliding_window;}
    void SetSlidingWindow(const int sliding_window){_sliding_window=sliding_window;}

    // ALiBi linear position bias (MPT, BLOOM): adds slope_h*(k-q) to each score
    // before the softmax, with a per-head slope that is a geometric sequence in
    // the head index. Replaces rotary/learned position encoding (when on, RoPE
    // should be off). Defaults to false, which disables it (no position bias).
    bool UseAlibi()const{return _use_alibi;}
    void SetUseAlibi(const bool use_alibi){_use_alibi=use_alibi;}

  protected:

  private:
    uint32_t _dim;
    uint32_t _num_heads;
    uint32_t _num_kv_heads;
    uint32_t _head_dim;
    bool _causal;
    bool _use_rope;
    float _rope_base;
    float _dropout_rate;
    int _rope_style;
    int _rope_dim;
    float _qk_norm_eps;
    float _attn_logit_softcap;
    int _sliding_window;
    bool _use_alibi;
};

}//end instance namespace
