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
// CAIF_DeviceLinearHeadConfig constructor. input_dim, output_dim and use_bias
// come from the caller; output_scale takes its no-op default (1.0) here in the
// initializer list.
//------------------------------------------------------------------------------
#include "caif_device_linear_head_config.h"

namespace instance
{

CAIF_DeviceLinearHeadConfig::CAIF_DeviceLinearHeadConfig(
    const uint32_t input_dim,
    const uint32_t output_dim,
    const bool use_bias):_input_dim(input_dim),
                         _output_dim(output_dim),
                         _use_bias(use_bias),
                         _output_scale(1.0f)
{
}

}//end instance namespace
