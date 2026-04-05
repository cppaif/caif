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
// Device-resident MoE Transformer Block implementation
//------------------------------------------------------------------------------
#include "caif_device_moe_block.h"
#include "caif_device_ops.h"
#include "caif_exception.h"
#include <cstdint>

using namespace instance;

CAIF_DeviceMoEBlock::CAIF_DeviceMoEBlock(const Config_t &config,
                                       CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                               _config(config),
                                                               _norm1(nullptr),
                                                               _attention(nullptr),
                                                               _norm2(nullptr),
                                                               _moe(nullptr),
                                                               _last_aux_losses({0.0f,0.0f})
{
  try
  {
    if(config.dim==0)
    {
      THROW_CAIFE("MoEBlock: dim must be > 0");
    }
    if(config.num_heads==0)
    {
      THROW_CAIFE("MoEBlock: num_heads must be > 0");
    }
    if(config.num_kv_heads==0)
    {
      THROW_CAIFE("MoEBlock: num_kv_heads must be > 0");
    }
    if(config.dim%config.num_heads!=0)
    {
      THROW_CAIFE("MoEBlock: dim must be divisible by num_heads");
    }
    if(config.num_experts==0)
    {
      THROW_CAIFE("MoEBlock: num_experts must be > 0");
    }
    if(config.top_k==0||config.top_k>config.num_experts)
    {
      THROW_CAIFE("MoEBlock: top_k must be > 0 and <= num_experts");
    }

    // Build sub-layers
    _norm1=std::make_unique<CAIF_DeviceRMSNorm>(config.dim,stream);

    CAIF_DeviceMultiHeadAttention::AttentionConfig_t attn_config;
    attn_config.dim=config.dim;
    attn_config.num_heads=config.num_heads;
    attn_config.num_kv_heads=config.num_kv_heads;
    attn_config.head_dim=config.dim/config.num_heads;
    attn_config.causal=config.causal;
    attn_config.use_rope=config.use_rope;
    attn_config.rope_base=config.rope_base;
    attn_config.dropout_rate=config.dropout_rate;
    _attention=std::make_unique<CAIF_DeviceMultiHeadAttention>(attn_config,stream);

    _norm2=std::make_unique<CAIF_DeviceRMSNorm>(config.dim,stream);

    // Build MoE layer
    CAIF_DeviceMoELayer::Config_t moe_config;
    moe_config.input_dim=config.dim;
    moe_config.hidden_dim=config.expert_ffn_dim;
    moe_config.num_experts=config.num_experts;
    moe_config.top_k=config.top_k;
    moe_config.expert_use_gated=config.expert_use_gated;
    moe_config.expert_use_bias=false;
    moe_config.num_shared_experts=config.num_shared_experts;
    if(config.shared_ffn_dim>0)
    {
      moe_config.shared_hidden_dim=config.shared_ffn_dim;
    }
    else
    {
      moe_config.shared_hidden_dim=config.expert_ffn_dim;
    }
    moe_config.fine_grained=false;
    moe_config.fine_grained_factor=1;
    moe_config.router_use_bias=false;
    moe_config.router_noise_std=config.router_noise_std;
    moe_config.capacity_factor=config.capacity_factor;
    moe_config.overflow_strategy=config.overflow_strategy;
    moe_config.balance_loss_weight=config.balance_loss_weight;
    moe_config.z_loss_weight=config.z_loss_weight;

    _moe=std::make_unique<CAIF_DeviceMoELayer>(moe_config,stream);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceMoEBlock::CAIF_DeviceMoEBlock(
  CAIF_DeviceMoEBlock &&other):CAIF_DeviceLayer(std::move(other)),
                              _config(other._config),
                              _norm1(std::move(other._norm1)),
                              _attention(std::move(other._attention)),
                              _norm2(std::move(other._norm2)),
                              _moe(std::move(other._moe)),
                              _cached_input(std::move(other._cached_input)),
                              _cached_after_attn(std::move(other._cached_after_attn)),
                              _last_aux_losses(other._last_aux_losses)
{
}

CAIF_DeviceMoEBlock &CAIF_DeviceMoEBlock::operator=(CAIF_DeviceMoEBlock &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _config=other._config;
      _norm1=std::move(other._norm1);
      _attention=std::move(other._attention);
      _norm2=std::move(other._norm2);
      _moe=std::move(other._moe);
      _cached_input=std::move(other._cached_input);
      _cached_after_attn=std::move(other._cached_after_attn);
      _last_aux_losses=other._last_aux_losses;
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoEBlock::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("MoEBlock: layer has been moved from");
    }

    // Pre-norm residual 1: h = x + attention(norm1(x))
    CAIF_DeviceTensor norm1_out=_norm1->Forward(input,training);
    CAIF_DeviceTensor attn_out=_attention->Forward(norm1_out,training);
    CAIF_DeviceTensor h=CAIF_DeviceTensor::Uninitialized(input.Shape(),*_stream);
    CAIF_DeviceOps::Add(input,attn_out,h);

    // Pre-norm residual 2: out = h + moe(norm2(h))
    CAIF_DeviceTensor norm2_out=_norm2->Forward(h,training);
    CAIF_DeviceMoELayer::MoEOutput_t moe_out=_moe->ForwardMoE(norm2_out,training);

    // Store auxiliary losses
    _last_aux_losses.balance_loss=moe_out.balance_loss;
    _last_aux_losses.z_loss=moe_out.z_loss;

    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(h.Shape(),*_stream);
    CAIF_DeviceOps::Add(h,moe_out.output,output);

    return output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoEBlock::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("MoEBlock: layer has been moved from");
    }

    // Backward through residual 2: out = h + moe(norm2(h))
    CAIF_DeviceTensor d_norm2_out=_moe->Backward(grad_output);
    CAIF_DeviceTensor d_h_from_moe=_norm2->Backward(d_norm2_out);
    CAIF_DeviceTensor d_h=CAIF_DeviceTensor::Uninitialized(grad_output.Shape(),*_stream);
    CAIF_DeviceOps::Add(grad_output,d_h_from_moe,d_h);

    // Backward through residual 1: h = x + attention(norm1(x))
    CAIF_DeviceTensor d_norm1_out=_attention->Backward(d_h);
    CAIF_DeviceTensor d_x_from_attn=_norm1->Backward(d_norm1_out);
    CAIF_DeviceTensor d_x=CAIF_DeviceTensor::Uninitialized(d_h.Shape(),*_stream);
    CAIF_DeviceOps::Add(d_h,d_x_from_attn,d_x);

    return d_x;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceMoEBlock::ZeroGradients()
{
  try
  {
    _norm1->ZeroGradients();
    _attention->ZeroGradients();
    _norm2->ZeroGradients();
    _moe->ZeroGradients();
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceMoEBlock::ParameterTensorCount()const
{
  try
  {
    return _norm1->ParameterTensorCount()+
           _attention->ParameterTensorCount()+
           _norm2->ParameterTensorCount()+
           _moe->ParameterTensorCount();
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceMoEBlock::SubLayerMapping_t CAIF_DeviceMoEBlock::MapIndex(size_t index)const
{
  SubLayerMapping_t mapping;
  const size_t norm1_count=_norm1->ParameterTensorCount();
  const size_t attn_count=_attention->ParameterTensorCount();
  const size_t norm2_count=_norm2->ParameterTensorCount();

  if(index<norm1_count)
  {
    mapping.sub_layer_idx=0;
    mapping.local_idx=index;
    return mapping;
  }
  index-=norm1_count;

  if(index<attn_count)
  {
    mapping.sub_layer_idx=1;
    mapping.local_idx=index;
    return mapping;
  }
  index-=attn_count;

  if(index<norm2_count)
  {
    mapping.sub_layer_idx=2;
    mapping.local_idx=index;
    return mapping;
  }
  index-=norm2_count;

  const size_t moe_count=_moe->ParameterTensorCount();
  if(index<moe_count)
  {
    mapping.sub_layer_idx=3;
    mapping.local_idx=index;
    return mapping;
  }

  THROW_CAIFE("MoEBlock::MapIndex: index out of range");
}

CAIF_DeviceTensor &CAIF_DeviceMoEBlock::ParameterTensor(size_t index)
{
  try
  {
    const auto mapping=MapIndex(index);
    if(mapping.sub_layer_idx==0)
    {
      return _norm1->ParameterTensor(mapping.local_idx);
    }
    if(mapping.sub_layer_idx==1)
    {
      return _attention->ParameterTensor(mapping.local_idx);
    }
    if(mapping.sub_layer_idx==2)
    {
      return _norm2->ParameterTensor(mapping.local_idx);
    }
    return _moe->ParameterTensor(mapping.local_idx);
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceMoEBlock::ParameterTensor(size_t index)const
{
  try
  {
    const auto mapping=MapIndex(index);
    if(mapping.sub_layer_idx==0)
    {
      return _norm1->ParameterTensor(mapping.local_idx);
    }
    if(mapping.sub_layer_idx==1)
    {
      return _attention->ParameterTensor(mapping.local_idx);
    }
    if(mapping.sub_layer_idx==2)
    {
      return _norm2->ParameterTensor(mapping.local_idx);
    }
    return _moe->ParameterTensor(mapping.local_idx);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceMoEBlock::GradientTensor(size_t index)
{
  try
  {
    const auto mapping=MapIndex(index);
    if(mapping.sub_layer_idx==0)
    {
      return _norm1->GradientTensor(mapping.local_idx);
    }
    if(mapping.sub_layer_idx==1)
    {
      return _attention->GradientTensor(mapping.local_idx);
    }
    if(mapping.sub_layer_idx==2)
    {
      return _norm2->GradientTensor(mapping.local_idx);
    }
    return _moe->GradientTensor(mapping.local_idx);
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceMoEBlock::GradientTensor(size_t index)const
{
  try
  {
    const auto mapping=MapIndex(index);
    if(mapping.sub_layer_idx==0)
    {
      return _norm1->GradientTensor(mapping.local_idx);
    }
    if(mapping.sub_layer_idx==1)
    {
      return _attention->GradientTensor(mapping.local_idx);
    }
    if(mapping.sub_layer_idx==2)
    {
      return _norm2->GradientTensor(mapping.local_idx);
    }
    return _moe->GradientTensor(mapping.local_idx);
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceMoEBlock::TotalParameterCount()const
{
  try
  {
    return _norm1->TotalParameterCount()+
           _attention->TotalParameterCount()+
           _norm2->TotalParameterCount()+
           _moe->TotalParameterCount();
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceMoEBlock::Description()const
{
  try
  {
    std::string causal_str;
    if(_config.causal==true)
    {
      causal_str="true";
    }
    else
    {
      causal_str="false";
    }
    return "MoEBlock(dim="+std::to_string(_config.dim)+
           ",heads="+std::to_string(_config.num_heads)+
           ",experts="+std::to_string(_config.num_experts)+
           ",top_k="+std::to_string(_config.top_k)+
           ",causal="+causal_str+")";
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceMoEBlock::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;

    auto norm1_names=_norm1->ParameterNames(prefix+"input_layernorm.");
    names.insert(names.end(),norm1_names.begin(),norm1_names.end());

    auto attn_names=_attention->ParameterNames(prefix+"self_attn.");
    names.insert(names.end(),attn_names.begin(),attn_names.end());

    auto norm2_names=_norm2->ParameterNames(prefix+"post_attention_layernorm.");
    names.insert(names.end(),norm2_names.begin(),norm2_names.end());

    auto moe_names=_moe->ParameterNames(prefix+"moe.");
    names.insert(names.end(),moe_names.begin(),moe_names.end());

    return names;
  }
  CAIF_CATCH_BLOCK()
}
