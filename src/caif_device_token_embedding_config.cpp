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
// CAIF_DeviceTokenEmbeddingConfig constructor. vocab_size and dim come from the
// caller; output_scale takes its no-op default (1.0) here in the initializer
// list.
//------------------------------------------------------------------------------
#include "caif_device_token_embedding_config.h"

namespace instance
{

CAIF_DeviceTokenEmbeddingConfig::CAIF_DeviceTokenEmbeddingConfig(
    const uint32_t vocab_size,
    const uint32_t dim):_vocab_size(vocab_size),
                        _dim(dim),
                        _output_scale(1.0f)
{
}

}//end instance namespace
