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
// CAIF_DeviceMLAttentionConfig constructor. The ten required fields come from
// the caller; rope_style takes its documented default and the cached-decode
// absorption threshold is computed from the model shape (see the crossover note).
//------------------------------------------------------------------------------
#include "caif_device_ml_attention_config.h"

namespace instance
{

// The cached-decode dispatch threshold is left 0 ("auto") here — the device
// layer resolves it at EnableKVCache from the model shape and this GPU's
// compute:bandwidth ratio (queried device properties; no hardware constant
// belongs in the config, which has no device access). A non-zero value set via
// SetDecodeAbsorbThreshold overrides the auto resolution.
CAIF_DeviceMLAttentionConfig::CAIF_DeviceMLAttentionConfig(const uint32_t dim,
                                                           const uint32_t num_heads,
                                                           const uint32_t q_lora_rank,
                                                           const uint32_t kv_lora_rank,
                                                           const uint32_t qk_rope_head_dim,
                                                           const uint32_t qk_nope_head_dim,
                                                           const uint32_t v_head_dim,
                                                           const bool causal,
                                                           const float rope_base,
                                                           const float rms_norm_eps):_dim(dim),
                                                                                     _num_heads(num_heads),
                                                                                     _q_lora_rank(q_lora_rank),
                                                                                     _kv_lora_rank(kv_lora_rank),
                                                                                     _qk_rope_head_dim(qk_rope_head_dim),
                                                                                     _qk_nope_head_dim(qk_nope_head_dim),
                                                                                     _v_head_dim(v_head_dim),
                                                                                     _causal(causal),
                                                                                     _rope_base(rope_base),
                                                                                     _rms_norm_eps(rms_norm_eps),
                                                                                     _rope_style(0),
                                                                                     _decode_absorb_threshold(0u)
{
}

}//end instance namespace
