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
// CAIF_DeviceTransformerModelConfig constructor. The required, architecture-
// defining fields come from the caller; the optional fields take their auto /
// no-op defaults here in the initializer list.
//------------------------------------------------------------------------------
#include "caif_device_transformer_model_config.h"

namespace instance
{

CAIF_DeviceTransformerModelConfig::CAIF_DeviceTransformerModelConfig(
    const uint32_t vocab_size,
    const uint32_t max_seq_len,
    const uint32_t dim,
    const uint32_t num_heads,
    const uint32_t num_layers,
    const CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e &pe_mode,
    const bool causal,
    const bool use_rope,
    const bool tie_weights):_vocab_size(vocab_size),
                            _max_seq_len(max_seq_len),
                            _dim(dim),
                            _num_heads(num_heads),
                            _num_kv_heads(0),
                            _num_layers(num_layers),
                            _ffn_dim(0),
                            _causal(causal),
                            _use_rope(use_rope),
                            _rope_style(0),
                            _pe_mode(pe_mode),
                            _output_dim(0),
                            _tie_weights(tie_weights),
                            _embed_scale(1.0f),
                            _logit_scale(1.0f)
{
}

}//end instance namespace
