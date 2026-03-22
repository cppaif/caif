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
// Device-resident generic Feed-Forward Network layer
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_FFN_H
#define CAIF_DEVICE_FFN_H

#include "caif_device_layer.h"
#include "caif_device_activation.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Generic FFN layer with pluggable activation
 *
 * Accepts any CAIF_DeviceActivation object. Queries IsGated() at construction
 * to determine parameter layout:
 *
 * Pointwise mode (2 weight tensors, no biases):
 *   hidden = input @ W1       -> [N, ffn_dim]
 *   act    = activation(hidden)
 *   output = act @ W2         -> [N, dim]
 *
 * Gated mode (3 weight tensors, no biases):
 *   gate   = input @ W_gate   -> [N, ffn_dim]
 *   up     = input @ W_up     -> [N, ffn_dim]
 *   act    = activation(gate, up)
 *   output = act @ W_down     -> [N, dim]
 *
 * Input: [batch, seq_len, dim] (3D) or [batch, dim] (2D).
 * Output: same shape as input.
 */
class CAIF_DeviceFFN:public CAIF_DeviceLayer
{
  public:
    struct FFNConfig_t
    {
      uint32_t dim;
      uint32_t ffn_dim;
    };

    struct FFNProjections_t
    {
      std::unique_ptr<CAIF_DeviceLayer> gate;
      std::unique_ptr<CAIF_DeviceLayer> up;
      std::unique_ptr<CAIF_DeviceLayer> down;
    };

    CAIF_DeviceFFN(const FFNConfig_t &config,
                  std::unique_ptr<CAIF_DeviceActivation> activation,
                  CAIF_CudaStream &stream);
    CAIF_DeviceFFN(const FFNConfig_t &config,
                  FFNProjections_t projections,
                  std::unique_ptr<CAIF_DeviceActivation> activation,
                  CAIF_CudaStream &stream);
    ~CAIF_DeviceFFN()override=default;

    // Move
    CAIF_DeviceFFN(CAIF_DeviceFFN &&other);
    CAIF_DeviceFFN &operator=(CAIF_DeviceFFN &&other);

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
    const FFNConfig_t &Config()const{return _config;}
    bool IsGated()const{return _is_gated;}

    // Weight initialization (Xavier uniform)
    void InitializeWeights(uint32_t seed=0);

  protected:

  private:
    FFNConfig_t _config;
    std::unique_ptr<CAIF_DeviceActivation> _activation;
    bool _is_gated;
    FFNProjections_t _projections;
    bool _use_projections;

    // Pointwise parameters (used when _is_gated==false)
    CAIF_DeviceTensor _w1;
    CAIF_DeviceTensor _w2;
    CAIF_DeviceTensor _grad_w1;
    CAIF_DeviceTensor _grad_w2;

    // Gated parameters (used when _is_gated==true)
    CAIF_DeviceTensor _w_gate;
    CAIF_DeviceTensor _w_up;
    CAIF_DeviceTensor _w_down;
    CAIF_DeviceTensor _grad_w_gate;
    CAIF_DeviceTensor _grad_w_up;
    CAIF_DeviceTensor _grad_w_down;

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_pre_activation;
    CAIF_DeviceTensor _cached_post_activation;
    CAIF_DeviceTensor _cached_gate_input;
    CAIF_DeviceTensor _cached_up_input;
    CAIF_DeviceTensor _cached_act_output;
    std::vector<uint32_t> _cached_input_shape;
};

}//end instance namespace

#endif  // CAIF_DEVICE_FFN_H
