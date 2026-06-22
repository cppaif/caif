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
// CAIF_MoEComposerModelConfig constructor. The eight defining fields come from
// the caller; storage_dtype and compute_dtype take their Float32 defaults here
// in the initializer list.
//------------------------------------------------------------------------------
#include "caif_moe_composer_model_config.h"

namespace instance
{

CAIF_MoEComposerModelConfig::CAIF_MoEComposerModelConfig(
    const uint32_t vocab_size,
    const uint32_t max_seq_len,
    const uint32_t output_dim,
    const bool tie_weights,
    const CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e &pe_mode,
    const float final_norm_eps,
    const uint32_t num_layers,
    const CAIF_MoEComposerBlockConfig &block_template):_vocab_size(vocab_size),
                                                       _max_seq_len(max_seq_len),
                                                       _output_dim(output_dim),
                                                       _tie_weights(tie_weights),
                                                       _pe_mode(pe_mode),
                                                       _final_norm_eps(final_norm_eps),
                                                       _num_layers(num_layers),
                                                       _block_template(block_template),
                                                       _storage_dtype(CAIF_DataType::CAIF_DataType_e::Float32),
                                                       _compute_dtype(CAIF_DataType::CAIF_DataType_e::Float32)
{
}

}//end instance namespace
