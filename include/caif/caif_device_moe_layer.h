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
// Device-resident MoE Layer (complete Mixture of Experts layer)
//------------------------------------------------------------------------------
#ifndef CAIF_DEVICE_MOE_LAYER_H
#define CAIF_DEVICE_MOE_LAYER_H

#include "caif_device_layer.h"
#include "caif_device_moe_router.h"
#include "caif_device_moe_expert.h"
#include "caif_device_tensor.h"
#include "caif_cuda_stream.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace instance
{

/**
 * @brief Complete Mixture of Experts Layer
 *
 * Combines router and multiple expert FFNs with sparse activation.
 * Only top-k experts are computed per token.
 *
 * Architecture:
 *   Input [batch, seq_len, dim]
 *     |
 *   Router -> expert_indices, expert_weights
 *     |
 *   Dispatch tokens to selected experts
 *     |
 *   Expert FFNs (parallel)
 *     |
 *   Combine expert outputs with routing weights
 *     |
 *   Output [batch, seq_len, dim]
 *
 * Auxiliary losses (for training stability):
 *   - Load balancing loss: Encourages uniform expert utilization
 *   - Router z-loss: Stabilizes router logits
 */
class CAIF_DeviceMoELayer:public CAIF_DeviceLayer
{
  public:

    enum class OverflowStrategy_e:uint8_t
    {
      Drop=0,         // Drop tokens that exceed capacity
      NoOp=1,         // Pass through unchanged
      Redistribute=2  // Route to next-best expert (Phase 3)
    };

    struct Config_t
    {
      // Dimensions
      uint32_t input_dim;           // Input/output dimension
      uint32_t hidden_dim;          // Expert FFN hidden dimension
      uint32_t num_experts;         // Total number of routed experts
      uint32_t top_k;               // Experts per token

      // Expert configuration
      bool expert_use_gated;        // Use gated FFN for experts
      bool expert_use_bias;         // Add bias to expert layers

      // Shared experts (DeepSeekMoE style)
      uint32_t num_shared_experts;  // Always-active experts (0 = none)
      uint32_t shared_hidden_dim;   // Hidden dim for shared experts (0 = use hidden_dim)

      // Fine-grained experts
      bool fine_grained;            // Use smaller, more numerous experts
      uint32_t fine_grained_factor; // Multiplier for expert count (e.g., 4 = 4x experts at 1/4 dim)

      // Router configuration
      bool router_use_bias;         // Add bias to router
      float router_noise_std;       // Noise for exploration (training)

      // Capacity and overflow
      float capacity_factor;        // Expert capacity multiplier (1.0 = exact)
      OverflowStrategy_e overflow_strategy;

      // Auxiliary losses
      float balance_loss_weight;    // Load balancing loss coefficient
      float z_loss_weight;          // Router z-loss coefficient
    };

    struct MoEOutput_t
    {
      CAIF_DeviceTensor output;          // [batch, seq_len, dim]
      float balance_loss;               // Load balancing auxiliary loss
      float z_loss;                     // Router z-loss
      CAIF_DeviceTensor expert_counts;   // [num_experts] tokens per expert
    };

    CAIF_DeviceMoELayer(const Config_t &config,CAIF_CudaStream &stream);
    CAIF_DeviceMoELayer(const Config_t &config,
                       std::vector<std::unique_ptr<CAIF_DeviceMoEExpert>> routed_experts,
                       std::vector<std::unique_ptr<CAIF_DeviceMoEExpert>> shared_experts,
                       CAIF_CudaStream &stream);
    ~CAIF_DeviceMoELayer()override=default;

    // Move
    CAIF_DeviceMoELayer(CAIF_DeviceMoELayer &&other);
    CAIF_DeviceMoELayer &operator=(CAIF_DeviceMoELayer &&other);

    // Main forward with auxiliary outputs
    MoEOutput_t ForwardMoE(const CAIF_DeviceTensor &input,bool training);

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
    uint32_t NumExperts()const{return _config.num_experts;}
    uint32_t NumSharedExperts()const{return _config.num_shared_experts;}
    uint32_t TopK()const{return _config.top_k;}
    bool IsFineGrained()const{return _config.fine_grained;}
    const CAIF_DeviceMoERouter &Router()const{return *_router;}
    const CAIF_DeviceMoEExpert &Expert(size_t index)const{return *_experts[index];}
    const CAIF_DeviceMoEExpert &SharedExpert(size_t index)const{return *_shared_experts[index];}

  protected:

  private:

    // Dispatch tokens to experts
    // Input: [num_tokens, dim], indices: [num_tokens, top_k]
    // Output: vector of [tokens_for_expert_i, dim] per expert
    std::vector<CAIF_DeviceTensor> DispatchTokens(const CAIF_DeviceTensor &input,
                                                  const CAIF_DeviceTensor &expert_indices,
                                                  std::vector<uint32_t> &token_counts);

    // Combine expert outputs back to original token positions
    // expert_outputs: vector of [tokens_for_expert_i, dim]
    // expert_weights: [num_tokens, top_k]
    // Output: [num_tokens, dim]
    CAIF_DeviceTensor CombineOutputs(const std::vector<CAIF_DeviceTensor> &expert_outputs,
                                    const CAIF_DeviceTensor &expert_indices,
                                    const CAIF_DeviceTensor &expert_weights,
                                    const std::vector<uint32_t> &token_counts,
                                    uint32_t num_tokens);

    // Compute auxiliary losses
    float ComputeBalanceLoss(const CAIF_DeviceTensor &router_probs,
                             const CAIF_DeviceTensor &expert_indices);
    float ComputeZLoss(const CAIF_DeviceTensor &router_logits);

    Config_t _config;

    // Sub-components
    std::unique_ptr<CAIF_DeviceMoERouter> _router;
    std::vector<std::unique_ptr<CAIF_DeviceMoEExpert>> _experts;
    std::vector<std::unique_ptr<CAIF_DeviceMoEExpert>> _shared_experts;  // Always-active experts

    // Cached for backward
    CAIF_DeviceTensor _cached_input;
    CAIF_DeviceMoERouter::RouterOutput_t _cached_router_output;
    std::vector<CAIF_DeviceTensor> _cached_expert_inputs;
    std::vector<CAIF_DeviceTensor> _cached_expert_outputs;
    std::vector<CAIF_DeviceTensor> _cached_shared_outputs;  // Shared expert outputs
    std::vector<uint32_t> _cached_token_counts;

    // Dispatch/combine index buffers
    CAIF_DeviceTensor _dispatch_indices;   // [num_tokens * top_k]
    CAIF_DeviceTensor _dispatch_weights;   // [num_tokens * top_k]

    // Overflow handling
    std::vector<uint32_t> _overflow_tokens;  // Tokens that exceeded capacity (for NoOp strategy)
};

}//end instance namespace

#endif  // CAIF_DEVICE_MOE_LAYER_H
