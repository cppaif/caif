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
// CAIF_DevicePooling2DConfig constructor — all four fields are required and
// come straight from the caller.
//------------------------------------------------------------------------------
#include "caif_device_pooling2d_config.h"

namespace instance
{

CAIF_DevicePooling2DConfig::CAIF_DevicePooling2DConfig(
    const uint32_t pool_height,
    const uint32_t pool_width,
    const uint32_t stride_height,
    const uint32_t stride_width):_pool_height(pool_height),
                                 _pool_width(pool_width),
                                 _stride_height(stride_height),
                                 _stride_width(stride_width)
{
}

}//end instance namespace
