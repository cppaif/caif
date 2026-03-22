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
// Device-resident MoE Router implementation
//------------------------------------------------------------------------------
#include "caif_device_moe_router.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include <random>
#include <cmath>

using namespace instance;

CAIF_DeviceMoERouter::CAIF_DeviceMoERouter(const Config_t &config,CAIF_CudaStream &stream)
  :CAIF_DeviceLayer(stream)
  ,_config(config)
{
  try
  {
    // Validate config
    if(_config.input_dim==0)
    {
      THROW_CAIFE("MoERouter: input_dim must be > 0");
    }
    if(_config.num_experts==0)
    {
      THROW_CAIFE("MoERouter: num_experts must be > 0");
    }
    if(_config.top_k==0||_config.top_k>_config.num_experts)
    {
      THROW_CAIFE("MoERouter: top_k must be > 0 and <= num_experts");
    }

    // Allocate router weights [input_dim, num_experts]
    _w_router=CAIF_DeviceTensor::Uninitialized({_config.input_dim,_config.num_experts},stream);
    _grad_w_router=CAIF_DeviceTensor::Zeros({_config.input_dim,_config.num_experts},stream);

    if(_config.use_bias==true)
    {
      _b_router=CAIF_DeviceTensor::Zeros({_config.num_experts},stream);
      _grad_b_router=CAIF_DeviceTensor::Zeros({_config.num_experts},stream);
    }

    // Xavier initialization for router weights
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale=std::sqrt(2.0f/static_cast<float>(_config.input_dim+_config.num_experts));
    std::normal_distribution<float> dist(0.0f,scale);

    std::vector<float> data(_config.input_dim*_config.num_experts);
    for(size_t i=0;i<data.size();++i)
    {
      data[i]=dist(gen);
    }
    _w_router.CopyFromHost(data.data(),data.size());

    _stream->Synchronize();
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceMoERouter::CAIF_DeviceMoERouter(CAIF_DeviceMoERouter &&other)
  :CAIF_DeviceLayer(std::move(other))
  ,_config(other._config)
  ,_w_router(std::move(other._w_router))
  ,_b_router(std::move(other._b_router))
  ,_grad_w_router(std::move(other._grad_w_router))
  ,_grad_b_router(std::move(other._grad_b_router))
  ,_cached_input(std::move(other._cached_input))
  ,_cached_logits(std::move(other._cached_logits))
  ,_cached_probs(std::move(other._cached_probs))
  ,_cached_indices(std::move(other._cached_indices))
{
}

CAIF_DeviceMoERouter &CAIF_DeviceMoERouter::operator=(CAIF_DeviceMoERouter &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _w_router=std::move(other._w_router);
    _b_router=std::move(other._b_router);
    _grad_w_router=std::move(other._grad_w_router);
    _grad_b_router=std::move(other._grad_b_router);
    _cached_input=std::move(other._cached_input);
    _cached_logits=std::move(other._cached_logits);
    _cached_probs=std::move(other._cached_probs);
    _cached_indices=std::move(other._cached_indices);
  }
  return *this;
}

CAIF_DeviceMoERouter::RouterOutput_t CAIF_DeviceMoERouter::Route(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    // Input can be [batch, seq_len, dim] or [num_tokens, dim]
    // Flatten to [num_tokens, dim] for processing
    const auto &shape=input.Shape();
    uint32_t num_tokens=0;
    CAIF_DeviceTensor flat_input;

    if(shape.size()==2)
    {
      num_tokens=shape[0];
      flat_input=input.Clone();
    }
    else if(shape.size()==3)
    {
      num_tokens=shape[0]*shape[1];
      flat_input=input.Clone();
      flat_input.Reshape({num_tokens,shape[2]});
    }
    else
    {
      THROW_CAIFE("MoERouter::Route: expected input shape [N, dim] or [batch, seq, dim]");
    }

    if(flat_input.Shape()[1]!=_config.input_dim)
    {
      THROW_CAIFE("MoERouter::Route: input dimension mismatch");
    }

    // Cache input for backward
    if(training==true)
    {
      _cached_input=flat_input.Clone();
    }

    // Compute router logits: input @ w_router + b_router
    // [num_tokens, input_dim] @ [input_dim, num_experts] = [num_tokens, num_experts]
    CAIF_DeviceTensor logits=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.num_experts},*_stream);
    CAIF_DeviceOps::MatMul(flat_input,_w_router,logits);

    if(_config.use_bias==true)
    {
      CAIF_DeviceOps::AddBias(logits,_b_router,logits);
    }

    // Add noise during training for exploration
    if(training==true&&_config.noise_std>0.0f)
    {
      // Generate noise on host and upload
      std::vector<float> noise_data(num_tokens*_config.num_experts);
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<float> dist(0.0f,_config.noise_std);
      for(size_t i=0;i<noise_data.size();++i)
      {
        noise_data[i]=dist(gen);
      }
      CAIF_DeviceTensor noise=CAIF_DeviceTensor::FromHostData(noise_data.data(),
                                                             {num_tokens,_config.num_experts},*_stream);
      CAIF_DeviceOps::Add(logits,noise,logits);
    }

    if(training==true)
    {
      _cached_logits=logits.Clone();
    }

    // Softmax over experts (for TopK and Soft) or over tokens (for ExpertChoice)
    CAIF_DeviceTensor probs=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.num_experts},*_stream);

    if(_config.routing_type==RoutingType_e::ExpertChoice)
    {
      // Expert Choice: softmax over tokens for each expert
      // Transpose logits to [num_experts, num_tokens], softmax, transpose back
      CAIF_DeviceTensor logits_t=CAIF_DeviceTensor::Uninitialized({_config.num_experts,num_tokens},*_stream);
      CAIF_DeviceOps::Transpose(logits,logits_t);
      CAIF_DeviceTensor probs_t=CAIF_DeviceTensor::Uninitialized({_config.num_experts,num_tokens},*_stream);
      CAIF_DeviceOps::Softmax(logits_t,probs_t);
      CAIF_DeviceOps::Transpose(probs_t,probs);
    }
    else
    {
      // TopK and Soft: softmax over experts for each token
      CAIF_DeviceOps::Softmax(logits,probs);
    }

    if(training==true)
    {
      _cached_probs=probs.Clone();
    }

    RouterOutput_t output;
    output.router_logits=logits.Clone();
    output.router_probs=probs.Clone();

    // Route based on routing type
    switch(_config.routing_type)
    {
      case RoutingType_e::TopK:
      {
        // Standard top-k: each token selects top_k experts
        CAIF_DeviceTensor indices=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.top_k},*_stream);
        CAIF_DeviceTensor weights=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.top_k},*_stream);
        CAIF_DeviceOps::TopK(probs,_config.top_k,indices,weights);
        CAIF_DeviceOps::NormalizeRows(weights,weights);

        if(training==true)
        {
          _cached_indices=indices.Clone();
        }

        output.expert_indices=std::move(indices);
        output.expert_weights=std::move(weights);
        break;
      }

      case RoutingType_e::ExpertChoice:
      {
        // Expert Choice: each expert selects top_k tokens
        // Capacity per expert = (num_tokens * top_k) / num_experts
        const uint32_t capacity=std::max(1u,(num_tokens*_config.top_k)/_config.num_experts);

        // Transpose probs to [num_experts, num_tokens] for expert-wise top-k
        CAIF_DeviceTensor probs_t=CAIF_DeviceTensor::Uninitialized({_config.num_experts,num_tokens},*_stream);
        CAIF_DeviceOps::Transpose(probs,probs_t);

        // Each expert selects top-capacity tokens
        CAIF_DeviceTensor token_indices=CAIF_DeviceTensor::Uninitialized({_config.num_experts,capacity},*_stream);
        CAIF_DeviceTensor token_weights=CAIF_DeviceTensor::Uninitialized({_config.num_experts,capacity},*_stream);
        CAIF_DeviceOps::TopK(probs_t,capacity,token_indices,token_weights);

        // Normalize weights per expert
        CAIF_DeviceOps::NormalizeRows(token_weights,token_weights);

        if(training==true)
        {
          _cached_indices=token_indices.Clone();
        }

        // For Expert Choice, indices are [num_experts, capacity] - which tokens each expert processes
        // This is different from TopK where indices are [num_tokens, top_k]
        output.expert_indices=std::move(token_indices);
        output.expert_weights=std::move(token_weights);
        break;
      }

      case RoutingType_e::Soft:
      {
        // Soft MoE: all experts process all tokens with softmax weights
        // No sparse selection - return full probability matrix
        // indices: just sequential expert indices for each token
        CAIF_DeviceTensor indices=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.num_experts},*_stream);
        std::vector<float> idx_data(num_tokens*_config.num_experts);
        for(uint32_t t=0;t<num_tokens;++t)
        {
          for(uint32_t e=0;e<_config.num_experts;++e)
          {
            idx_data[t*_config.num_experts+e]=static_cast<float>(e);
          }
        }
        indices.CopyFromHost(idx_data.data(),idx_data.size());

        if(training==true)
        {
          _cached_indices=indices.Clone();
        }

        output.expert_indices=std::move(indices);
        output.expert_weights=probs.Clone();  // Full softmax probabilities
        break;
      }

      default:
        THROW_CAIFE("MoERouter::Route: unknown routing type");
    }

    return output;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoERouter::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    // Forward returns the routing weights for compatibility with layer interface
    RouterOutput_t output=Route(input,training);
    return std::move(output.expert_weights);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoERouter::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    // This backward is for the standard layer interface
    // grad_output: gradient w.r.t. expert_weights [num_tokens, top_k]
    return BackwardRouting(grad_output);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoERouter::BackwardRouting(const CAIF_DeviceTensor &grad_weights)
{
  try
  {
    // grad_weights: [num_tokens, top_k] - gradient w.r.t. normalized routing weights
    const auto &shape=grad_weights.Shape();
    const uint32_t num_tokens=shape[0];

    // First, we need to backprop through the normalization
    // weights_normalized = weights / sum(weights)
    // This is complex because we selected top-k

    // For simplicity in Phase 1, we backprop directly through softmax
    // assuming the top-k selection doesn't affect gradients significantly
    // (the gradients for non-selected experts are 0)

    // Scatter grad_weights back to full expert dimension
    CAIF_DeviceTensor grad_probs=CAIF_DeviceTensor::Zeros({num_tokens,_config.num_experts},*_stream);
    CAIF_DeviceOps::ScatterAdd(grad_weights,_cached_indices,grad_probs);

    // Backward through softmax
    // grad_logits = probs * (grad_probs - sum(grad_probs * probs, axis=-1, keepdim=True))
    CAIF_DeviceTensor grad_logits=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.num_experts},*_stream);
    CAIF_DeviceOps::SoftmaxBackward(_cached_probs,grad_probs,grad_logits);

    // Backward through linear projection
    // grad_w_router += input^T @ grad_logits
    CAIF_DeviceTensor grad_w_batch=
      CAIF_DeviceTensor::Uninitialized({_config.input_dim,_config.num_experts},*_stream);
    CAIF_DeviceOps::MatMulTransposeA(_cached_input,grad_logits,grad_w_batch);
    CAIF_DeviceOps::Add(_grad_w_router,grad_w_batch,_grad_w_router);

    if(_config.use_bias==true)
    {
      // grad_b_router += sum(grad_logits, axis=0)
      CAIF_DeviceTensor grad_b_batch=CAIF_DeviceTensor::Uninitialized({_config.num_experts},*_stream);
      CAIF_DeviceOps::SumAxis(grad_logits,0,grad_b_batch);
      CAIF_DeviceOps::Add(_grad_b_router,grad_b_batch,_grad_b_router);
    }

    // grad_input = grad_logits @ w_router^T
    CAIF_DeviceTensor grad_input=CAIF_DeviceTensor::Uninitialized({num_tokens,_config.input_dim},*_stream);
    CAIF_DeviceOps::MatMulTransposeB(grad_logits,_w_router,grad_input);

    return grad_input;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceMoERouter::ZeroGradients()
{
  try
  {
    _grad_w_router.Fill(0.0f);
    if(_config.use_bias==true)
    {
      _grad_b_router.Fill(0.0f);
    }
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceMoERouter::ParameterTensorCount()const
{
  size_t count=1;  // w_router
  if(_config.use_bias==true)
  {
    count+=1;  // b_router
  }
  return count;
}

CAIF_DeviceTensor &CAIF_DeviceMoERouter::ParameterTensor(size_t index)
{
  if(index==0)
  {
    return _w_router;
  }
  if(_config.use_bias==true&&index==1)
  {
    return _b_router;
  }
  THROW_CAIFE("MoERouter::ParameterTensor: index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceMoERouter::ParameterTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoERouter*>(this)->ParameterTensor(index);
}

CAIF_DeviceTensor &CAIF_DeviceMoERouter::GradientTensor(size_t index)
{
  if(index==0)
  {
    return _grad_w_router;
  }
  if(_config.use_bias==true&&index==1)
  {
    return _grad_b_router;
  }
  THROW_CAIFE("MoERouter::GradientTensor: index out of range");
}

const CAIF_DeviceTensor &CAIF_DeviceMoERouter::GradientTensor(size_t index)const
{
  return const_cast<CAIF_DeviceMoERouter*>(this)->GradientTensor(index);
}

size_t CAIF_DeviceMoERouter::TotalParameterCount()const
{
  size_t count=_config.input_dim*_config.num_experts;  // w_router
  if(_config.use_bias==true)
  {
    count+=_config.num_experts;  // b_router
  }
  return count;
}

std::string CAIF_DeviceMoERouter::Description()const
{
  std::string desc="MoERouter[";
  desc+=std::to_string(_config.input_dim)+"->"+std::to_string(_config.num_experts);
  desc+=",top_k="+std::to_string(_config.top_k);
  if(_config.use_bias==true)
  {
    desc+=",bias";
  }
  if(_config.noise_std>0.0f)
  {
    desc+=",noise="+std::to_string(_config.noise_std);
  }
  desc+="]";
  return desc;
}

std::vector<std::string> CAIF_DeviceMoERouter::ParameterNames(const std::string &prefix)const
{
  std::vector<std::string> names;
  names.push_back(prefix+"w_router");
  if(_config.use_bias==true)
  {
    names.push_back(prefix+"b_router");
  }
  return names;
}
