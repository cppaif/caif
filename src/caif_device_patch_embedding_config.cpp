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
// CAIF_DevicePatchEmbeddingConfig constructor — all six fields are required and
// come straight from the caller.
//------------------------------------------------------------------------------
#include "caif_device_patch_embedding_config.h"

namespace instance
{

CAIF_DevicePatchEmbeddingConfig::CAIF_DevicePatchEmbeddingConfig(
    const uint32_t image_height,
    const uint32_t image_width,
    const uint32_t channels,
    const uint32_t patch_size,
    const uint32_t dim,
    const bool use_cls_token):_image_height(image_height),
                              _image_width(image_width),
                              _channels(channels),
                              _patch_size(patch_size),
                              _dim(dim),
                              _use_cls_token(use_cls_token)
{
}

}//end instance namespace
