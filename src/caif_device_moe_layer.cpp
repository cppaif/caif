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
// Device-resident MoE Layer implementation
//------------------------------------------------------------------------------
#include "caif_device_moe_layer.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include <cmath>

using namespace instance;

CAIF_DeviceMoELayer::CAIF_DeviceMoELayer(const Config_t &config,CAIF_CudaStream &stream)
  :CAIF_DeviceLayer(stream)
  ,_config(config)
{
  try
  {
    // Validate config
    if(_config.input_dim==0)
    {
      THROW_CAIFE("MoELayer: input_dim must be > 0");
    }
    if(_config.hidden_dim==0)
    {
      THROW_CAIFE("MoELayer: hidden_dim must be > 0");
    }
    if(_config.num_experts==0)
    {
      THROW_CAIFE("MoELayer: num_experts must be > 0");
    }
    if(_config.top_k==0||_config.top_k>_config.num_experts)
    {
      THROW_CAIFE("MoELayer: top_k must be > 0 and <= num_experts");
    }

    // Compute actual expert dimensions for fine-grained mode
    uint32_t actual_num_experts=_config.num_experts;
    uint32_t actual_hidden_dim=_config.hidden_dim;

    if(_config.fine_grained&&_config.fine_grained_factor>1)
    {
      // Fine-grained: more experts with smaller hidden dimensions
      // E.g., factor=4 means 4x experts with 1/4 hidden dim each
      actual_num_experts=_config.num_experts*_config.fine_grained_factor;
      actual_hidden_dim=_config.hidden_dim/_config.fine_grained_factor;
      if(actual_hidden_dim==0)
      {
        actual_hidden_dim=1;  // Minimum hidden dim
      }
    }

    // Create router (routes to all routed experts)
    CAIF_DeviceMoERouter::Config_t router_config;
    router_config.input_dim=_config.input_dim;
    router_config.num_experts=actual_num_experts;
    router_config.top_k=_config.top_k;
    router_config.routing_type=CAIF_DeviceMoERouter::RoutingType_e::TopK;
    router_config.use_bias=_config.router_use_bias;
    router_config.noise_std=_config.router_noise_std;

    _router=std::make_unique<CAIF_DeviceMoERouter>(router_config,stream);

    // Create routed experts
    CAIF_DeviceMoEExpert::Config_t expert_config;
    expert_config.input_dim=_config.input_dim;
    expert_config.hidden_dim=actual_hidden_dim;
    expert_config.use_gated=_config.expert_use_gated;
    expert_config.use_bias=_config.expert_use_bias;

    _experts.reserve(actual_num_experts);
    for(uint32_t i=0;i<actual_num_experts;++i)
    {
      _experts.push_back(std::make_unique<CAIF_DeviceMoEExpert>(expert_config,stream));
    }

    // Create shared experts (DeepSeekMoE style - always active for all tokens)
    if(_config.num_shared_experts>0)
    {
      CAIF_DeviceMoEExpert::Config_t shared_config;
      shared_config.input_dim=_config.input_dim;
      if(_config.shared_hidden_dim>0)
      {
        shared_config.hidden_dim=_config.shared_hidden_dim;
      }
      else
      {
        shared_config.hidden_dim=_config.hidden_dim;
      }
      shared_config.use_gated=_config.expert_use_gated;
      shared_config.use_bias=_config.expert_use_bias;

      _shared_experts.reserve(_config.num_shared_experts);
      for(uint32_t i=0;i<_config.num_shared_experts;++i)
      {
        _shared_experts.push_back(std::make_unique<CAIF_DeviceMoEExpert>(shared_config,stream));
      }
    }

    _stream->Synchronize();
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceMoELayer::CAIF_DeviceMoELayer(const Config_t &config,
                                       std::vector<std::unique_ptr<CAIF_DeviceMoEExpert>> routed_experts,
                                       std::vector<std::unique_ptr<CAIF_DeviceMoEExpert>> shared_experts,
                                       CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                                _config(config)
{
  try
  {
    if(_config.input_dim==0)
    {
      THROW_CAIFE("MoELayer: input_dim must be > 0");
    }
    if(_config.num_experts==0)
    {
      THROW_CAIFE("MoELayer: num_experts must be > 0");
    }
    if(_config.top_k==0||_config.top_k>_config.num_experts)
    {
      THROW_CAIFE("MoELayer: top_k must be > 0 and <= num_experts");
    }

    // Compute actual number of routed experts for router
    uint32_t actual_num_experts=_config.num_experts;
    if(_config.fine_grained&&_config.fine_grained_factor>1)
    {
      actual_num_experts=_config.num_experts*_config.fine_grained_factor;
    }

    // Create router
    CAIF_DeviceMoERouter::Config_t router_config;
    router_config.input_dim=_config.input_dim;
    router_config.num_experts=actual_num_experts;
    router_config.top_k=_config.top_k;
    router_config.routing_type=CAIF_DeviceMoERouter::RoutingType_e::TopK;
    router_config.use_bias=_config.router_use_bias;
    router_config.noise_std=_config.router_noise_std;

    _router=std::make_unique<CAIF_DeviceMoERouter>(router_config,stream);

    // Take ownership of caller-provided experts
    _experts=std::move(routed_experts);
    _shared_experts=std::move(shared_experts);

    _stream->Synchronize();
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceMoELayer::CAIF_DeviceMoELayer(CAIF_DeviceMoELayer &&other)
  :CAIF_DeviceLayer(std::move(other))
  ,_config(other._config)
  ,_router(std::move(other._router))
  ,_experts(std::move(other._experts))
  ,_shared_experts(std::move(other._shared_experts))
  ,_cached_input(std::move(other._cached_input))
  ,_cached_router_output(std::move(other._cached_router_output))
  ,_cached_expert_inputs(std::move(other._cached_expert_inputs))
  ,_cached_expert_outputs(std::move(other._cached_expert_outputs))
  ,_cached_shared_outputs(std::move(other._cached_shared_outputs))
  ,_cached_token_counts(std::move(other._cached_token_counts))
  ,_dispatch_indices(std::move(other._dispatch_indices))
  ,_dispatch_weights(std::move(other._dispatch_weights))
{
}

CAIF_DeviceMoELayer &CAIF_DeviceMoELayer::operator=(CAIF_DeviceMoELayer &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _router=std::move(other._router);
    _experts=std::move(other._experts);
    _shared_experts=std::move(other._shared_experts);
    _cached_input=std::move(other._cached_input);
    _cached_router_output=std::move(other._cached_router_output);
    _cached_expert_inputs=std::move(other._cached_expert_inputs);
    _cached_expert_outputs=std::move(other._cached_expert_outputs);
    _cached_shared_outputs=std::move(other._cached_shared_outputs);
    _cached_token_counts=std::move(other._cached_token_counts);
    _dispatch_indices=std::move(other._dispatch_indices);
    _dispatch_weights=std::move(other._dispatch_weights);
  }
  return *this;
}

CAIF_DeviceMoELayer::MoEOutput_t CAIF_DeviceMoELayer::ForwardMoE(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    // Input: [batch, seq_len, dim] or [num_tokens, dim]
    const auto &shape=input.Shape();
    uint32_t num_tokens=0;
    uint32_t batch_size=1;
    uint32_t seq_len=1;

    // Clone input for potential reshape
    CAIF_DeviceTensor flat_input=input.Clone();

    if(shape.size()==2)
    {
      num_tokens=shape[0];
    }
    else if(shape.size()==3)
    {
      batch_size=shape[0];
      seq_len=shape[1];
      num_tokens=batch_size*seq_len;
      flat_input.Reshape({num_tokens,shape[2]});
    }
    else
    {
      THROW_CAIFE("MoELayer::ForwardMoE: expected input shape [N, dim] or [batch, seq, dim]");
    }

    if(flat_input.Shape()[1]!=_config.input_dim)
    {
      THROW_CAIFE("MoELayer::ForwardMoE: input dimension mismatch");
    }

    // Cache input for backward
    if(training==true)
    {
      _cached_input=flat_input.Clone();
    }

    // Route tokens to experts
    CAIF_DeviceMoERouter::RouterOutput_t router_output=_router->Route(flat_input,training);

    if(training==true)
    {
      _cached_router_output=std::move(router_output);
      // Need to re-route to get fresh output (router_output was moved)
      router_output=_router->Route(flat_input,false);
    }

    // Dispatch tokens to experts
    std::vector<CAIF_DeviceTensor> expert_inputs=DispatchTokens(flat_input,
                                                                router_output.expert_indices,
                                                                _cached_token_counts);

    // Run each expert on its assigned tokens
    std::vector<CAIF_DeviceTensor> expert_outputs;
    expert_outputs.reserve(_config.num_experts);

    if(training==true)
    {
      _cached_expert_inputs.clear();
      _cached_expert_inputs.reserve(_config.num_experts);
      _cached_expert_outputs.clear();
      _cached_expert_outputs.reserve(_config.num_experts);
    }

    for(uint32_t i=0;i<_config.num_experts;++i)
    {
      if(_cached_token_counts[i]>0)
      {
        CAIF_DeviceTensor expert_out=_experts[i]->Forward(expert_inputs[i],training);
        expert_outputs.push_back(std::move(expert_out));

        if(training==true)
        {
          _cached_expert_inputs.push_back(expert_inputs[i].Clone());
          _cached_expert_outputs.push_back(expert_outputs.back().Clone());
        }
      }
      else
      {
        // No tokens for this expert
        expert_outputs.push_back(CAIF_DeviceTensor());
        if(training==true)
        {
          _cached_expert_inputs.push_back(CAIF_DeviceTensor());
          _cached_expert_outputs.push_back(CAIF_DeviceTensor());
        }
      }
    }

    // Combine expert outputs back to original positions
    CAIF_DeviceTensor combined=CombineOutputs(expert_outputs,
                                              router_output.expert_indices,
                                              router_output.expert_weights,
                                              _cached_token_counts,
                                              num_tokens);

    // Run shared experts on ALL tokens and add to output (DeepSeekMoE style)
    if(_shared_experts.size()>0)
    {
      if(training==true)
      {
        _cached_shared_outputs.clear();
        _cached_shared_outputs.reserve(_shared_experts.size());
      }

      for(size_t i=0;i<_shared_experts.size();++i)
      {
        CAIF_DeviceTensor shared_out=_shared_experts[i]->Forward(flat_input,training);

        if(training==true)
        {
          _cached_shared_outputs.push_back(shared_out.Clone());
        }

        // Add shared expert output to combined output
        // Output = routed_output + shared_output (for each shared expert)
        CAIF_DeviceOps::Add(combined,shared_out,combined);
      }
    }

    // Reshape back to original shape if needed
    if(shape.size()==3)
    {
      combined.Reshape({batch_size,seq_len,_config.input_dim});
    }

    // Compute auxiliary losses
    float balance_loss=0.0f;
    float z_loss=0.0f;

    if(training==true&&_config.balance_loss_weight>0.0f)
    {
      balance_loss=ComputeBalanceLoss(_cached_router_output.router_probs,
                                       _cached_router_output.expert_indices);
    }

    if(training==true&&_config.z_loss_weight>0.0f)
    {
      z_loss=ComputeZLoss(_cached_router_output.router_logits);
    }

    // Compute expert token counts
    CAIF_DeviceTensor expert_counts=CAIF_DeviceTensor::Uninitialized({_config.num_experts},*_stream);
    std::vector<float> counts_host(_config.num_experts);
    for(uint32_t i=0;i<_config.num_experts;++i)
    {
      counts_host[i]=static_cast<float>(_cached_token_counts[i]);
    }
    expert_counts.CopyFromHost(counts_host.data(),counts_host.size());

    MoEOutput_t moe_output;
    moe_output.output=std::move(combined);
    moe_output.balance_loss=balance_loss;
    moe_output.z_loss=z_loss;
    moe_output.expert_counts=std::move(expert_counts);

    return moe_output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoELayer::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    MoEOutput_t output=ForwardMoE(input,training);
    return std::move(output.output);
  }
  CAIF_CATCH_BLOCK()
}

std::vector<CAIF_DeviceTensor> CAIF_DeviceMoELayer::DispatchTokens(const CAIF_DeviceTensor &input,
                                                                  const CAIF_DeviceTensor &expert_indices,
                                                                  std::vector<uint32_t> &token_counts)
{
  try
  {
    // input: [num_tokens, dim]
    // expert_indices: [num_tokens, top_k]
    const uint32_t num_tokens=input.Shape()[0];
    const uint32_t dim=input.Shape()[1];

    // Compute expert capacity
    // capacity = (tokens_per_batch / num_experts) * capacity_factor * top_k
    const uint32_t capacity=static_cast<uint32_t>(
      std::ceil(static_cast<float>(num_tokens)/_config.num_experts*_config.capacity_factor*_config.top_k));

    // Count tokens per expert from indices (respecting capacity based on strategy)
    token_counts.resize(_config.num_experts,0);

    // Copy indices to host for counting
    std::vector<float> indices_float(num_tokens*_config.top_k);
    expert_indices.CopyToHost(indices_float.data());

    // Track which tokens overflow (for NoOp and Redistribute strategies)
    _overflow_tokens.clear();

    // First pass: count tokens per expert with overflow handling
    for(uint32_t t=0;t<num_tokens;++t)
    {
      for(uint32_t k=0;k<_config.top_k;++k)
      {
        const int32_t expert_idx=static_cast<int32_t>(indices_float[t*_config.top_k+k]);
        if(expert_idx>=0&&expert_idx<static_cast<int32_t>(_config.num_experts))
        {
          if(token_counts[expert_idx]<capacity)
          {
            ++token_counts[expert_idx];
          }
          else
          {
            // Handle overflow based on strategy
            switch(_config.overflow_strategy)
            {
              case OverflowStrategy_e::Drop:
                // Token is simply not processed by this expert
                break;

              case OverflowStrategy_e::NoOp:
                // Track overflow token for passthrough
                _overflow_tokens.push_back(t);
                break;

              case OverflowStrategy_e::Redistribute:
                // Try to find next-best expert with capacity
                // Look at remaining experts in the top_k list
                for(uint32_t k2=k+1;k2<_config.top_k;++k2)
                {
                  const int32_t alt_idx=static_cast<int32_t>(indices_float[t*_config.top_k+k2]);
                  if(alt_idx>=0&&alt_idx<static_cast<int32_t>(_config.num_experts))
                  {
                    if(token_counts[alt_idx]<capacity)
                    {
                      ++token_counts[alt_idx];
                      break;
                    }
                  }
                }
                break;
            }
          }
        }
      }
    }

    // Allocate expert input tensors based on actual counts
    std::vector<CAIF_DeviceTensor> expert_inputs;
    expert_inputs.reserve(_config.num_experts);

    for(uint32_t i=0;i<_config.num_experts;++i)
    {
      if(token_counts[i]>0)
      {
        expert_inputs.push_back(CAIF_DeviceTensor::Uninitialized({token_counts[i],dim},*_stream));
      }
      else
      {
        expert_inputs.push_back(CAIF_DeviceTensor());
      }
    }

    // Use CAIF_DeviceOps for actual dispatch
    CAIF_DeviceOps::MoEDispatch(input,expert_indices,_config.top_k,token_counts,expert_inputs);

    return expert_inputs;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoELayer::CombineOutputs(const std::vector<CAIF_DeviceTensor> &expert_outputs,
                                                     const CAIF_DeviceTensor &expert_indices,
                                                     const CAIF_DeviceTensor &expert_weights,
                                                     const std::vector<uint32_t> &token_counts,
                                                     uint32_t num_tokens)
{
  try
  {
    // Combine expert outputs back to original token positions with routing weights
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Zeros({num_tokens,_config.input_dim},*_stream);

    // Use CAIF_DeviceOps for actual combine
    CAIF_DeviceOps::MoECombine(expert_outputs,
                              expert_indices,
                              expert_weights,
                              _config.top_k,
                              token_counts,
                              output);

    return output;
  }
  CAIF_CATCH_BLOCK()
}

float CAIF_DeviceMoELayer::ComputeBalanceLoss(const CAIF_DeviceTensor &router_probs,
                                              const CAIF_DeviceTensor &/*expert_indices*/)
{
  try
  {
    // Load balancing loss: L_balance = α * n * Σᵢ(fᵢ * Pᵢ)
    // fᵢ = fraction of tokens routed to expert i
    // Pᵢ = mean routing probability for expert i

    const uint32_t num_tokens=router_probs.Shape()[0];

    // Compute fraction of tokens per expert
    std::vector<float> fractions(_config.num_experts,0.0f);
    float total_assignments=static_cast<float>(num_tokens*_config.top_k);

    for(uint32_t i=0;i<_config.num_experts;++i)
    {
      fractions[i]=static_cast<float>(_cached_token_counts[i])/total_assignments;
    }

    // Compute mean probability per expert
    std::vector<float> mean_probs(_config.num_experts,0.0f);
    CAIF_DeviceTensor sum_probs=CAIF_DeviceTensor::Uninitialized({_config.num_experts},*_stream);
    CAIF_DeviceOps::SumAxis(router_probs,0,sum_probs);

    std::vector<float> sum_probs_host(_config.num_experts);
    sum_probs.CopyToHost(sum_probs_host.data());

    for(uint32_t i=0;i<_config.num_experts;++i)
    {
      mean_probs[i]=sum_probs_host[i]/static_cast<float>(num_tokens);
    }

    // Compute loss
    float loss=0.0f;
    for(uint32_t i=0;i<_config.num_experts;++i)
    {
      loss+=fractions[i]*mean_probs[i];
    }

    return _config.balance_loss_weight*static_cast<float>(_config.num_experts)*loss;
  }
  CAIF_CATCH_BLOCK()
}

float CAIF_DeviceMoELayer::ComputeZLoss(const CAIF_DeviceTensor &router_logits)
{
  try
  {
    // Router z-loss: L_z = β * (1/n) * Σᵢ(log Σⱼ exp(rᵢⱼ))²
    // Stabilizes router logits by penalizing large values

    const uint32_t num_tokens=router_logits.Shape()[0];

    // Compute log-sum-exp per token
    CAIF_DeviceTensor logsumexp=CAIF_DeviceTensor::Uninitialized({num_tokens},*_stream);
    CAIF_DeviceOps::LogSumExp(router_logits,logsumexp);

    // Square the log-sum-exp values
    CAIF_DeviceTensor logsumexp_sq=CAIF_DeviceTensor::Uninitialized({num_tokens},*_stream);
    CAIF_DeviceOps::Multiply(logsumexp,logsumexp,logsumexp_sq);

    // Sum and average
    CAIF_DeviceTensor total=CAIF_DeviceTensor::Uninitialized({1},*_stream);
    CAIF_DeviceOps::Sum(logsumexp_sq,total);

    float total_host=0.0f;
    total.CopyToHost(&total_host);

    return _config.z_loss_weight*total_host/static_cast<float>(num_tokens);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoELayer::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    // grad_output: [batch, seq_len, dim] or [num_tokens, dim]
    const auto &shape=grad_output.Shape();
    uint32_t num_tokens=0;

    // Clone for potential reshape
    CAIF_DeviceTensor flat_grad=grad_output.Clone();

    if(shape.size()==2)
    {
      num_tokens=shape[0];
    }
    else if(shape.size()==3)
    {
      num_tokens=shape[0]*shape[1];
      flat_grad.Reshape({num_tokens,shape[2]});
    }
    else
    {
      THROW_CAIFE("MoELayer::Backward: expected grad shape [N, dim] or [batch, seq, dim]");
    }

    // Backward through combine: distribute gradients to experts
    std::vector<CAIF_DeviceTensor> grad_expert_outputs;
    CAIF_DeviceTensor grad_weights;

    CAIF_DeviceOps::MoECombineBackward(flat_grad,
                                       _cached_expert_outputs,
                                       _cached_router_output.expert_indices,
                                       _cached_router_output.expert_weights,
                                       _config.top_k,
                                       _cached_token_counts,
                                       grad_expert_outputs,
                                       grad_weights);

    // Backward through each expert
    std::vector<CAIF_DeviceTensor> grad_expert_inputs;
    grad_expert_inputs.reserve(_config.num_experts);

    for(uint32_t i=0;i<_config.num_experts;++i)
    {
      if(_cached_token_counts[i]>0)
      {
        CAIF_DeviceTensor grad_in=_experts[i]->Backward(grad_expert_outputs[i]);
        grad_expert_inputs.push_back(std::move(grad_in));
      }
      else
      {
        grad_expert_inputs.push_back(CAIF_DeviceTensor());
      }
    }

    // Backward through dispatch: combine gradients back to original positions
    CAIF_DeviceTensor grad_input_from_experts=CAIF_DeviceTensor::Zeros({num_tokens,_config.input_dim},*_stream);
    CAIF_DeviceOps::MoEDispatchBackward(grad_expert_inputs,
                                        _cached_router_output.expert_indices,
                                        _config.top_k,
                                        _cached_token_counts,
                                        grad_input_from_experts);

    // Backward through router
    CAIF_DeviceTensor grad_input_from_router=_router->BackwardRouting(grad_weights);

    // Combine gradients from routed experts and router
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
    CAIF_DeviceOps::Add(grad_input_from_experts,grad_input_from_router,grad_input);

    // Backward through shared experts
    // Shared experts receive the same gradient as the combined output (since output = routed + shared)
    if(_shared_experts.size()>0)
    {
      for(size_t i=0;i<_shared_experts.size();++i)
      {
        CAIF_DeviceTensor grad_from_shared=_shared_experts[i]->Backward(flat_grad);
        // Add gradient contribution from shared expert
        CAIF_DeviceOps::Add(grad_input,grad_from_shared,grad_input);
      }
    }

    // Reshape if needed
    if(shape.size()==3)
    {
      grad_input.Reshape({shape[0],shape[1],shape[2]});
    }
    return grad_input;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceMoELayer::ZeroGradients()
{
  try
  {
    _router->ZeroGradients();
    for(auto &expert:_experts)
    {
      expert->ZeroGradients();
    }
    for(auto &shared:_shared_experts)
    {
      shared->ZeroGradients();
    }
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceMoELayer::ParameterTensorCount()const
{
  size_t count=_router->ParameterTensorCount();
  for(const auto &expert:_experts)
  {
    count+=expert->ParameterTensorCount();
  }
  for(const auto &shared:_shared_experts)
  {
    count+=shared->ParameterTensorCount();
  }
  return count;
}

CAIF_DeviceTensor &CAIF_DeviceMoELayer::ParameterTensor(size_t index)
{
  // Router parameters first
  size_t router_count=_router->ParameterTensorCount();
  if(index<router_count)
  {
    return _router->ParameterTensor(index);
  }
  index-=router_count;

  // Then routed expert parameters
  for(auto &expert:_experts)
  {
    size_t expert_count=expert->ParameterTensorCount();
    if(index<expert_count)
    {
      return expert->ParameterTensor(index);
    }
    index-=expert_count;
  }

  // Then shared expert parameters
  for(auto &shared:_shared_experts)
  {
    size_t shared_count=shared->ParameterTensorCount();
    if(index<shared_count)
    {
      return shared->ParameterTensor(index);
    }
    index-=shared_count;
  }

  THROW_CAIFE("MoELayer::ParameterTensor: index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceMoELayer::ParameterTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoELayer*>(this)->ParameterTensor(index);
}

CAIF_DeviceTensor &CAIF_DeviceMoELayer::GradientTensor(size_t index)
{
  // Router gradients first
  size_t router_count=_router->ParameterTensorCount();
  if(index<router_count)
  {
    return _router->GradientTensor(index);
  }
  index-=router_count;

  // Then routed expert gradients
  for(auto &expert:_experts)
  {
    size_t expert_count=expert->ParameterTensorCount();
    if(index<expert_count)
    {
      return expert->GradientTensor(index);
    }
    index-=expert_count;
  }

  // Then shared expert gradients
  for(auto &shared:_shared_experts)
  {
    size_t shared_count=shared->ParameterTensorCount();
    if(index<shared_count)
    {
      return shared->GradientTensor(index);
    }
    index-=shared_count;
  }

  THROW_CAIFE("MoELayer::GradientTensor: index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceMoELayer::GradientTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoELayer*>(this)->GradientTensor(index);
}

size_t CAIF_DeviceMoELayer::TotalParameterCount()const
{
  size_t count=_router->TotalParameterCount();
  for(const auto &expert:_experts)
  {
    count+=expert->TotalParameterCount();
  }
  for(const auto &shared:_shared_experts)
  {
    count+=shared->TotalParameterCount();
  }
  return count;
}

std::string CAIF_DeviceMoELayer::Description()const
{
  std::string desc="MoELayer[";
  desc+=std::to_string(_config.input_dim)+"->"+std::to_string(_config.hidden_dim);
  desc+=",experts="+std::to_string(_experts.size());
  if(_config.num_shared_experts>0)
  {
    desc+=",shared="+std::to_string(_config.num_shared_experts);
  }
  desc+=",top_k="+std::to_string(_config.top_k);
  if(_config.expert_use_gated==true)
  {
    desc+=",gated";
  }
  if(_config.fine_grained==true)
  {
    desc+=",fine_grained("+std::to_string(_config.fine_grained_factor)+"x)";
  }
  desc+=",cap="+std::to_string(_config.capacity_factor);
  desc+="]";
  return desc;
}

std::vector<std::string> CAIF_DeviceMoELayer::ParameterNames(const std::string &prefix)const
{
  std::vector<std::string> names;

  // Router parameters
  std::vector<std::string> router_names=_router->ParameterNames(prefix+"router.");
  names.insert(names.end(),router_names.begin(),router_names.end());

  // Routed expert parameters
  for(size_t i=0;i<_experts.size();++i)
  {
    std::string expert_prefix=prefix+"expert_"+std::to_string(i)+".";
    std::vector<std::string> expert_names=_experts[i]->ParameterNames(expert_prefix);
    names.insert(names.end(),expert_names.begin(),expert_names.end());
  }

  // Shared expert parameters
  for(size_t i=0;i<_shared_experts.size();++i)
  {
    std::string shared_prefix=prefix+"shared_expert_"+std::to_string(i)+".";
    std::vector<std::string> shared_names=_shared_experts[i]->ParameterNames(shared_prefix);
    names.insert(names.end(),shared_names.begin(),shared_names.end());
  }

  return names;
}
