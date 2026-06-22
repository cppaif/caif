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
// CAIF_DeviceRelativePositionBiasConfig constructor — all four fields are
// required and come straight from the caller.
//------------------------------------------------------------------------------
#include "caif_device_relative_position_bias_config.h"

namespace instance
{

CAIF_DeviceRelativePositionBiasConfig::CAIF_DeviceRelativePositionBiasConfig(
    const uint32_t num_heads,
    const uint32_t num_buckets,
    const uint32_t max_distance,
    const bool bidirectional):_num_heads(num_heads),
                              _num_buckets(num_buckets),
                              _max_distance(max_distance),
                              _bidirectional(bidirectional)
{
}

}//end instance namespace
