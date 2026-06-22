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
// CAIF_DeviceMoEExpertConfig constructor — all four fields are required and
// come straight from the caller.
//------------------------------------------------------------------------------
#include "caif_device_moe_expert_config.h"

namespace instance
{

CAIF_DeviceMoEExpertConfig::CAIF_DeviceMoEExpertConfig(
    const uint32_t input_dim,
    const uint32_t hidden_dim,
    const bool use_gated,
    const bool use_bias):_input_dim(input_dim),
                         _hidden_dim(hidden_dim),
                         _use_gated(use_gated),
                         _use_bias(use_bias)
{
}

}//end instance namespace
