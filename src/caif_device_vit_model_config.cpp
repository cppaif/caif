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
// CAIF_DeviceViTModelConfig constructor. The twelve required fields come from
// the caller; rope_style takes its documented default (0) here in the
// initializer list.
//------------------------------------------------------------------------------
#include "caif_device_vit_model_config.h"

namespace instance
{

CAIF_DeviceViTModelConfig::CAIF_DeviceViTModelConfig(
    const uint32_t image_height,
    const uint32_t image_width,
    const uint32_t channels,
    const uint32_t patch_size,
    const uint32_t dim,
    const uint32_t num_layers,
    const uint32_t num_heads,
    const uint32_t ffn_hidden_dim,
    const float dropout_rate,
    const uint32_t num_classes,
    const bool use_rope,
    const float rope_base):_image_height(image_height),
                           _image_width(image_width),
                           _channels(channels),
                           _patch_size(patch_size),
                           _dim(dim),
                           _num_layers(num_layers),
                           _num_heads(num_heads),
                           _ffn_hidden_dim(ffn_hidden_dim),
                           _dropout_rate(dropout_rate),
                           _num_classes(num_classes),
                           _use_rope(use_rope),
                           _rope_base(rope_base),
                           _rope_style(0)
{
}

}//end instance namespace
