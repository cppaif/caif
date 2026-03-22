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
// LoRA (Low-Rank Adaptation) adapter wrapping any CAIF_DeviceLayer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_LORA_ADAPTER_H
#define CAIF_DEVICE_LORA_ADAPTER_H

#include "caif_device_layer.h"
#include "caif_constants.h"
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace instance
{

/**
 * @brief LoRA adapter that adds low-rank trainable parameters to any layer
 *
 * Wraps a base layer (typically CAIF_DeviceFrozenLinear) and adds
 * trainable LoRA A and B matrices. During forward:
 *   output = base_layer(input) + (input @ A^T @ B^T) * (alpha / rank)
 *
 * Only LoRA A and B are exposed as trainable parameters. The base layer's
 * parameters are NOT exposed to the optimizer.
 *
 * Initialization: A = Kaiming uniform, B = zeros.
 * This means LoRA initially produces zero output (identity behavior).
 */
class CAIF_DeviceLoRAAdapter:public CAIF_DeviceLayer
{
  public:
    struct LoRAConfig_t
    {
      uint32_t rank;
      float alpha;
      uint32_t input_dim;
      uint32_t output_dim;
    };

    CAIF_DeviceLoRAAdapter(const LoRAConfig_t &config,
                          std::unique_ptr<CAIF_DeviceLayer> base_layer,
                          CAIF_CudaStream &stream,
                          uint32_t seed=0);

    ~CAIF_DeviceLoRAAdapter()override=default;

    // Movable
    CAIF_DeviceLoRAAdapter(CAIF_DeviceLoRAAdapter &&other);
    CAIF_DeviceLoRAAdapter &operator=(CAIF_DeviceLoRAAdapter &&other);

    // --- CAIF_DeviceLayer interface ---
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

    // --- LoRA-specific ---
    const LoRAConfig_t &Config()const{return _config;}

  protected:

  private:
    LoRAConfig_t _config;
    std::unique_ptr<CAIF_DeviceLayer> _base_layer;

    // LoRA parameters: A=[rank, input_dim], B=[output_dim, rank]
    CAIF_DeviceTensor _lora_a;
    CAIF_DeviceTensor _lora_b;

    // Gradients
    CAIF_DeviceTensor _grad_lora_a;
    CAIF_DeviceTensor _grad_lora_b;

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_lora_hidden;
};

}//end instance namespace

#endif  // CAIF_DEVICE_LORA_ADAPTER_H
