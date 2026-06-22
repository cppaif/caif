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
// Configuration for CAIF_DeviceViTModel. The twelve architecture-defining
// fields (image extents, patch size, transformer dims/depth, dropout, class
// count, and RoPE on/off + base) are required by the constructor so a ViT can
// never be built half-configured. rope_style carries a documented default (set
// in the constructor's initializer list, not as an in-class initializer) and is
// adjusted through its setter.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DeviceViTModelConfig:public CAIF_Base
{
  public:
    // The twelve required, architecture-defining fields. rope_style (RoPE
    // layout) takes its documented default and is set via SetRopeStyle().
    CAIF_DeviceViTModelConfig(const uint32_t image_height,
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
                              const float rope_base);

    // Input image height in pixels.
    uint32_t ImageHeight()const{return _image_height;}
    void SetImageHeight(const uint32_t image_height){_image_height=image_height;}

    // Input image width in pixels.
    uint32_t ImageWidth()const{return _image_width;}
    void SetImageWidth(const uint32_t image_width){_image_width=image_width;}

    // Number of input channels.
    uint32_t Channels()const{return _channels;}
    void SetChannels(const uint32_t channels){_channels=channels;}

    // Square patch side length in pixels.
    uint32_t PatchSize()const{return _patch_size;}
    void SetPatchSize(const uint32_t patch_size){_patch_size=patch_size;}

    // Transformer (hidden) dimension.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Number of transformer blocks.
    uint32_t NumLayers()const{return _num_layers;}
    void SetNumLayers(const uint32_t num_layers){_num_layers=num_layers;}

    // Number of attention heads.
    uint32_t NumHeads()const{return _num_heads;}
    void SetNumHeads(const uint32_t num_heads){_num_heads=num_heads;}

    // FFN hidden dimension.
    uint32_t FfnHiddenDim()const{return _ffn_hidden_dim;}
    void SetFfnHiddenDim(const uint32_t ffn_hidden_dim){_ffn_hidden_dim=ffn_hidden_dim;}

    // Dropout rate.
    float DropoutRate()const{return _dropout_rate;}
    void SetDropoutRate(const float dropout_rate){_dropout_rate=dropout_rate;}

    // Number of classification output classes.
    uint32_t NumClasses()const{return _num_classes;}
    void SetNumClasses(const uint32_t num_classes){_num_classes=num_classes;}

    // Apply rotary position embeddings (instead of learned positions).
    bool UseRope()const{return _use_rope;}
    void SetUseRope(const bool use_rope){_use_rope=use_rope;}

    // RoPE base frequency (theta).
    float RopeBase()const{return _rope_base;}
    void SetRopeBase(const float rope_base){_rope_base=rope_base;}

    // RoPE layout: 0 = interleaved, 1 = half-split. Defaults to 0.
    int RopeStyle()const{return _rope_style;}
    void SetRopeStyle(const int rope_style){_rope_style=rope_style;}

  protected:

  private:
    uint32_t _image_height;
    uint32_t _image_width;
    uint32_t _channels;
    uint32_t _patch_size;
    uint32_t _dim;
    uint32_t _num_layers;
    uint32_t _num_heads;
    uint32_t _ffn_hidden_dim;
    float _dropout_rate;
    uint32_t _num_classes;
    bool _use_rope;
    float _rope_base;
    int _rope_style;
};

}//end instance namespace
