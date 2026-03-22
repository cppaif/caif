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
// Generic pre-norm residual block layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_PRE_NORM_BLOCK_H
#define CAIF_DEVICE_PRE_NORM_BLOCK_H

#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Generic pre-norm residual block.
 *
 * Composes an arbitrary number of (norm, layer) pairs with residual
 * connections. Each stage applies:
 *
 *   x = x + layer(norm(x))
 *
 * This single class replaces the need for architecture-specific block
 * classes (CAIF_DeviceTransformerBlock, CAIF_DeviceMoEBlock, etc.).
 *
 * The caller plugs in whatever combination of attention, FFN, MoE,
 * or any other CAIF_DeviceLayer they want:
 *
 * Standard transformer:  (RMSNorm, MHA)  + (RMSNorm, FFN)
 * MoE transformer:       (RMSNorm, MHA)  + (RMSNorm, MoE)
 * MLA + MoE (GLM-4.7):   (RMSNorm, MLA)  + (RMSNorm, MoE)
 * 3-stage block:          (Norm, CrossAttn) + (Norm, SelfAttn) + (Norm, FFN)
 */
class CAIF_DevicePreNormBlock:public CAIF_DeviceLayer
{
  public:

    struct SubLayer_t
    {
      std::string norm_prefix;
      std::string layer_prefix;
      std::unique_ptr<CAIF_DeviceLayer> norm;
      std::unique_ptr<CAIF_DeviceLayer> layer;
    };

    typedef std::vector<SubLayer_t> SubLayerVec_t;

    CAIF_DevicePreNormBlock(SubLayerVec_t sub_layers,CAIF_CudaStream &stream);
    ~CAIF_DevicePreNormBlock() override=default;

    // Move semantics
    CAIF_DevicePreNormBlock(CAIF_DevicePreNormBlock &&other);
    CAIF_DevicePreNormBlock &operator=(CAIF_DevicePreNormBlock &&other);

    // CAIF_DeviceLayer interface
    CAIF_DeviceTensor Forward(const CAIF_DeviceTensor &input,bool training) override;
    CAIF_DeviceTensor Backward(const CAIF_DeviceTensor &grad_output) override;
    void ZeroGradients() override;
    size_t ParameterTensorCount()const override;
    CAIF_DeviceTensor &ParameterTensor(size_t index) override;
    const CAIF_DeviceTensor &ParameterTensor(size_t index)const override;
    CAIF_DeviceTensor &GradientTensor(size_t index) override;
    const CAIF_DeviceTensor &GradientTensor(size_t index)const override;
    size_t TotalParameterCount()const override;
    std::string Description()const override;
    std::vector<std::string> ParameterNames(const std::string &prefix="")const override;

    size_t SubLayerCount()const{return _sub_layers.size();}

  protected:

  private:

    struct SubLayerMapping_t
    {
      size_t stage_idx;
      bool is_norm;
      size_t local_idx;
    };

    SubLayerMapping_t MapIndex(size_t index)const;
    CAIF_DeviceLayer &LayerByMapping(const SubLayerMapping_t &mapping);
    const CAIF_DeviceLayer &LayerByMapping(const SubLayerMapping_t &mapping)const;

    SubLayerVec_t _sub_layers;
};

}//end instance namespace

#endif  // CAIF_DEVICE_PRE_NORM_BLOCK_H
