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
// AIF - AI Framework
// Device-resident patch embedding layer (ViT-style)
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_PATCH_EMBEDDING_H
#define CAIF_DEVICE_PATCH_EMBEDDING_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Converts images into patch embedding sequences (ViT-style).
 *
 * Input:  [batch, height, width, channels] (BHWC)
 * Output: [batch, num_patches, dim] or [batch, num_patches+1, dim] (with CLS)
 *
 * Forward: extract_patches -> MatMul(W_proj) -> BiasAdd(b_proj) -> optional cls_prepend
 *
 * Parameters:
 *   0: W_proj     [patch_flat_dim, dim]   Xavier uniform
 *   1: b_proj     [dim]                   Zeros
 *   2: cls_token  [1, dim]               Xavier uniform (only with use_cls_token)
 */
class CAIF_DevicePatchEmbedding:public CAIF_DeviceLayer
{
  public:
    struct Config_t
    {
      uint32_t image_height;
      uint32_t image_width;
      uint32_t channels;
      uint32_t patch_size;
      uint32_t dim;
      bool use_cls_token;
    };

    CAIF_DevicePatchEmbedding(const Config_t &config,
                             CAIF_CudaStream &stream);
    ~CAIF_DevicePatchEmbedding()override=default;

    // Move
    CAIF_DevicePatchEmbedding(CAIF_DevicePatchEmbedding &&other);
    CAIF_DevicePatchEmbedding &operator=(CAIF_DevicePatchEmbedding &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training)override;
    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output)override;
    void ZeroGradients()override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index)override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index)override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    // Accessors
    uint32_t PatchSize()const{return _config.patch_size;}
    uint32_t Dim()const{return _config.dim;}
    uint32_t NumPatches()const{return _num_patches;}
    bool UseCLSToken()const{return _config.use_cls_token;}

  protected:

  private:
    Config_t _config;
    uint32_t _num_patches_h;
    uint32_t _num_patches_w;
    uint32_t _num_patches;
    uint32_t _patch_flat_dim;

    CAIF_DeviceTensor _w_proj;       // [patch_flat_dim, dim]
    CAIF_DeviceTensor _b_proj;       // [dim]
    CAIF_DeviceTensor _cls_token;    // [1, dim] (only when use_cls_token)

    CAIF_DeviceTensor _grad_w_proj;  // [patch_flat_dim, dim]
    CAIF_DeviceTensor _grad_b_proj;  // [dim]
    CAIF_DeviceTensor _grad_cls;     // [1, dim] (only when use_cls_token)

    // Cached for backward
    CAIF_DeviceTensor _cached_input;       // [batch, H, W, C]
    CAIF_DeviceTensor _cached_patches;     // [batch*num_patches, patch_flat_dim]
    uint32_t _cached_batch;
};

}//end instance namespace

#endif  // CAIF_DEVICE_PATCH_EMBEDDING_H
