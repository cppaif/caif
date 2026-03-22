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
// Device-resident MoE Router (expert selection network)
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_MOE_ROUTER_H
#define CAIF_DEVICE_MOE_ROUTER_H

#include "caif_device_layer.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include <cstdint>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Router network for Mixture of Experts
 *
 * Computes routing weights and selects top-k experts per token.
 *
 * Architecture:
 *   Input [batch, seq_len, dim] -> Linear [dim, num_experts] -> Softmax -> Top-k
 *
 * Outputs:
 *   - expert_indices: [batch, seq_len, top_k] - indices of selected experts
 *   - expert_weights: [batch, seq_len, top_k] - normalized weights for selected experts
 *   - router_logits:  [batch, seq_len, num_experts] - raw logits (for aux losses)
 *
 * Parameters:
 *   - W_router [dim, num_experts]
 *   - b_router [num_experts] (optional)
 */
class CAIF_DeviceMoERouter:public CAIF_DeviceLayer
{
  public:

    enum class RoutingType_e:uint8_t
    {
      TopK=0,           // Standard top-k routing
      ExpertChoice=1,   // Expert selects tokens (Phase 3)
      Soft=2            // Soft routing - all experts (Phase 3)
    };

    struct Config_t
    {
      uint32_t input_dim;       // Input dimension
      uint32_t num_experts;     // Total number of experts
      uint32_t top_k;           // Number of experts per token
      RoutingType_e routing_type;
      bool use_bias;            // Add bias to router projection
      float noise_std;          // Router noise for exploration (training only)
    };

    struct RouterOutput_t
    {
      CAIF_DeviceTensor expert_indices;   // [batch*seq_len, top_k] int32
      CAIF_DeviceTensor expert_weights;   // [batch*seq_len, top_k] float
      CAIF_DeviceTensor router_logits;    // [batch*seq_len, num_experts] float
      CAIF_DeviceTensor router_probs;     // [batch*seq_len, num_experts] float (softmax)
    };

    CAIF_DeviceMoERouter(const Config_t &config,CAIF_CudaStream &stream);
    ~CAIF_DeviceMoERouter()override=default;

    // Move
    CAIF_DeviceMoERouter(CAIF_DeviceMoERouter &&other);
    CAIF_DeviceMoERouter &operator=(CAIF_DeviceMoERouter &&other);

    // Main routing function
    RouterOutput_t Route(const CAIF_DeviceTensor &input,bool training);

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

    // Backward for routing (takes grad w.r.t. expert_weights)
    CAIF_DeviceTensor BackwardRouting(const CAIF_DeviceTensor &grad_weights);

    // Accessors
    const Config_t &Config()const{return _config;}
    uint32_t InputDim()const{return _config.input_dim;}
    uint32_t NumExperts()const{return _config.num_experts;}
    uint32_t TopK()const{return _config.top_k;}

  protected:

  private:

    Config_t _config;

    // Router weights
    CAIF_DeviceTensor _w_router;    // [input_dim, num_experts]
    CAIF_DeviceTensor _b_router;    // [num_experts] (optional)

    // Gradients
    CAIF_DeviceTensor _grad_w_router;
    CAIF_DeviceTensor _grad_b_router;

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceTensor _cached_logits;
    CAIF_DeviceTensor _cached_probs;
    CAIF_DeviceTensor _cached_indices;
};

}//end instance namespace

#endif  // CAIF_DEVICE_MOE_ROUTER_H
