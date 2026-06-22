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
// CAIF_DeviceMultiHeadAttentionConfig constructor. The eight required fields
// come from the caller; rope_style, rope_dim and qk_norm_eps take their
// documented defaults here in the initializer list.
//------------------------------------------------------------------------------
#include "caif_device_multi_head_attention_config.h"

namespace instance
{

CAIF_DeviceMultiHeadAttentionConfig::CAIF_DeviceMultiHeadAttentionConfig(
    const uint32_t dim,
    const uint32_t num_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim,
    const bool causal,
    const bool use_rope,
    const float rope_base,
    const float dropout_rate):_dim(dim),
                              _num_heads(num_heads),
                              _num_kv_heads(num_kv_heads),
                              _head_dim(head_dim),
                              _causal(causal),
                              _use_rope(use_rope),
                              _rope_base(rope_base),
                              _dropout_rate(dropout_rate),
                              _rope_style(g_caif_mha_default_rope_style),
                              _rope_dim(g_caif_mha_default_rope_dim),
                              _qk_norm_eps(g_caif_mha_default_qk_norm_eps),
                              _attn_logit_softcap(g_caif_mha_default_attn_logit_softcap),
                              _sliding_window(g_caif_mha_default_sliding_window),
                              _use_alibi(g_caif_mha_default_use_alibi)
{
}

}//end instance namespace
