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
// Device-resident MoE Expert (single expert FFN)
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_MOE_EXPERT_H
#define CAIF_DEVICE_MOE_EXPERT_H

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
 * @brief Single expert FFN for Mixture of Experts
 *
 * Standard FFN architecture: Linear -> Activation -> Linear
 * Can also support gated FFN (SwiGLU style): (Linear_gate * Activation(Linear_up)) -> Linear_down
 *
 * Input:  [num_tokens, input_dim]
 * Output: [num_tokens, input_dim]
 *
 * Parameters (standard FFN):
 *   - W_up   [input_dim, hidden_dim]
 *   - W_down [hidden_dim, input_dim]
 *
 * Parameters (gated FFN):
 *   - W_gate [input_dim, hidden_dim]
 *   - W_up   [input_dim, hidden_dim]
 *   - W_down [hidden_dim, input_dim]
 */
class CAIF_DeviceMoEExpert:public CAIF_DeviceLayer
{
  public:

    struct Config_t
    {
      uint32_t input_dim;      // Input/output dimension
      uint32_t hidden_dim;     // FFN hidden dimension
      bool use_gated;          // Use gated FFN (SwiGLU style)
      bool use_bias;           // Add bias to linear layers
    };

    struct MoEExpertProjections_t
    {
      std::unique_ptr<CAIF_DeviceLayer> gate;
      std::unique_ptr<CAIF_DeviceLayer> up;
      std::unique_ptr<CAIF_DeviceLayer> down;
    };

    CAIF_DeviceMoEExpert(const Config_t &config,CAIF_CudaStream &stream);
    CAIF_DeviceMoEExpert(const Config_t &config,
                        MoEExpertProjections_t projections,
                        CAIF_CudaStream &stream);
    ~CAIF_DeviceMoEExpert()override=default;

    // Move
    CAIF_DeviceMoEExpert(CAIF_DeviceMoEExpert &&other);
    CAIF_DeviceMoEExpert &operator=(CAIF_DeviceMoEExpert &&other);

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
    const Config_t &Config()const{return _config;}
    uint32_t InputDim()const{return _config.input_dim;}
    uint32_t HiddenDim()const{return _config.hidden_dim;}
    bool UseGated()const{return _config.use_gated;}

  protected:

  private:

    Config_t _config;
    MoEExpertProjections_t _projections;
    bool _use_projections;

    // Weights
    CAIF_DeviceTensor _w_gate;      // [input_dim, hidden_dim] (gated only)
    CAIF_DeviceTensor _w_up;        // [input_dim, hidden_dim]
    CAIF_DeviceTensor _w_down;      // [hidden_dim, input_dim]

    // Biases (optional)
    CAIF_DeviceTensor _b_gate;      // [hidden_dim] (gated only)
    CAIF_DeviceTensor _b_up;        // [hidden_dim]
    CAIF_DeviceTensor _b_down;      // [input_dim]

    // Gradients
    CAIF_DeviceTensor _grad_w_gate;
    CAIF_DeviceTensor _grad_w_up;
    CAIF_DeviceTensor _grad_w_down;
    CAIF_DeviceTensor _grad_b_gate;
    CAIF_DeviceTensor _grad_b_up;
    CAIF_DeviceTensor _grad_b_down;

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_gate_out;   // Before activation (gated only)
    CAIF_DeviceTensor _cached_up_out;     // Before activation
    CAIF_DeviceTensor _cached_hidden;     // After activation (or gate*activation for gated)
};

}//end instance namespace

#endif  // CAIF_DEVICE_MOE_EXPERT_H
