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
// CAIF_DeviceTransformerBlockConfig constructor. The eight required fields come
// from the caller; rope_style takes its documented default (0) here in the
// initializer list.
//------------------------------------------------------------------------------
#include "caif_device_transformer_block_config.h"

namespace instance
{

CAIF_DeviceTransformerBlockConfig::CAIF_DeviceTransformerBlockConfig(
    const uint32_t dim,
    const uint32_t num_heads,
    const uint32_t num_kv_heads,
    const uint32_t ffn_dim,
    const float dropout_rate,
    const bool causal,
    const bool use_rope,
    const float rope_base):_dim(dim),
                           _num_heads(num_heads),
                           _num_kv_heads(num_kv_heads),
                           _ffn_dim(ffn_dim),
                           _dropout_rate(dropout_rate),
                           _causal(causal),
                           _use_rope(use_rope),
                           _rope_base(rope_base),
                           _rope_style(0)
{
}

}//end instance namespace
