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
// Configuration for CAIF_DevicePatchEmbedding. All six fields (image extents,
// channels, patch size, embedding dim, CLS-token flag) are required by the
// constructor so the patch embedding can never be built half-configured.
//------------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "caif_base.h"

namespace instance
{

class CAIF_DevicePatchEmbeddingConfig:public CAIF_Base
{
  public:
    // All six fields are required: the input image height/width, channel count,
    // patch size, embedding dimension, and whether a CLS token is prepended.
    CAIF_DevicePatchEmbeddingConfig(const uint32_t image_height,
                                    const uint32_t image_width,
                                    const uint32_t channels,
                                    const uint32_t patch_size,
                                    const uint32_t dim,
                                    const bool use_cls_token);

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

    // Embedding dimension each patch is projected to.
    uint32_t Dim()const{return _dim;}
    void SetDim(const uint32_t dim){_dim=dim;}

    // Prepend a learnable CLS token to the patch sequence.
    bool UseCLSToken()const{return _use_cls_token;}
    void SetUseCLSToken(const bool use_cls_token){_use_cls_token=use_cls_token;}

  protected:

  private:
    uint32_t _image_height;
    uint32_t _image_width;
    uint32_t _channels;
    uint32_t _patch_size;
    uint32_t _dim;
    bool _use_cls_token;
};

}//end instance namespace
