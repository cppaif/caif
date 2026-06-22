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
// CAIF_DeviceCrossAttentionConfig constructor — all five fields are required
// and come straight from the caller.
//------------------------------------------------------------------------------
#include "caif_device_cross_attention_config.h"

namespace instance
{

CAIF_DeviceCrossAttentionConfig::CAIF_DeviceCrossAttentionConfig(
    const uint32_t dim,
    const uint32_t kv_input_dim,
    const uint32_t num_heads,
    const uint32_t num_kv_heads,
    const uint32_t head_dim):_dim(dim),
                             _kv_input_dim(kv_input_dim),
                             _num_heads(num_heads),
                             _num_kv_heads(num_kv_heads),
                             _head_dim(head_dim)
{
}

}//end instance namespace
