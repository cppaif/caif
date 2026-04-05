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

#include "caif_device_transformer_block.h"
#include "caif_device_ops.h"
#include "caif_device_gated_activations.h"
#include "caif_exception.h"
#include <cstdint>

using namespace instance;

CAIF_DeviceTransformerBlock::CAIF_DeviceTransformerBlock(const TransformerBlockConfig_t &config,
                                                       std::unique_ptr<CAIF_DeviceActivation> activation,
                                                       CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                                                                               _config(config),
                                                                               _effective_ffn_dim(0),
                                                                               _norm1(nullptr),
                                                                               _attention(nullptr),
                                                                               _norm2(nullptr),
                                                                               _ffn(nullptr)
{
  try
  {
    if(config.dim==0)
    {
      THROW_CAIFE("TransformerBlock: dim must be > 0");
    }
    if(config.num_heads==0)
    {
      THROW_CAIFE("TransformerBlock: num_heads must be > 0");
    }
    if(config.num_kv_heads==0)
    {
      THROW_CAIFE("TransformerBlock: num_kv_heads must be > 0");
    }
    if(config.dim%config.num_heads!=0)
    {
      THROW_CAIFE("TransformerBlock: dim must be divisible by num_heads");
    }

    // Compute effective FFN dim
    if(config.ffn_dim==0)
    {
      _effective_ffn_dim=ComputeDefaultFFNDim(config.dim);
    }
    else
    {
      _effective_ffn_dim=config.ffn_dim;
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

    CAIF_DeviceFFN::FFNConfig_t ffn_config;
    ffn_config.dim=config.dim;
    ffn_config.ffn_dim=_effective_ffn_dim;
    _ffn=std::make_unique<CAIF_DeviceFFN>(ffn_config,std::move(activation),stream);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTransformerBlock::CAIF_DeviceTransformerBlock(const TransformerBlockConfig_t &config,
                                                       CAIF_CudaStream &stream):CAIF_DeviceTransformerBlock(
                                                         config,
                                                         std::make_unique<CAIF_DeviceSwiGLUActivation>(),
                                                         stream)
{
}

CAIF_DeviceTransformerBlock::CAIF_DeviceTransformerBlock(
  CAIF_DeviceTransformerBlock &&other):CAIF_DeviceLayer(std::move(other)),
                                      _config(other._config),
                                      _effective_ffn_dim(other._effective_ffn_dim),
                                      _norm1(std::move(other._norm1)),
                                      _attention(std::move(other._attention)),
                                      _norm2(std::move(other._norm2)),
                                      _ffn(std::move(other._ffn))
{
}

CAIF_DeviceTransformerBlock &CAIF_DeviceTransformerBlock::operator=(CAIF_DeviceTransformerBlock &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _config=other._config;
      _effective_ffn_dim=other._effective_ffn_dim;
      _norm1=std::move(other._norm1);
      _attention=std::move(other._attention);
      _norm2=std::move(other._norm2);
      _ffn=std::move(other._ffn);
    }
    return *this;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceTransformerBlock::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("TransformerBlock: layer has been moved from");
    }

    // Pre-norm residual 1: h = x + attention(norm1(x))
    CAIF_DeviceTensor norm1_out=_norm1->Forward(input,training);
    CAIF_DeviceTensor attn_out=_attention->Forward(norm1_out,training);
    CAIF_DeviceTensor h=CAIF_DeviceTensor::Uninitialized(input.Shape(),*_stream);
    CAIF_DeviceOps::Add(input,attn_out,h);

    // Pre-norm residual 2: out = h + ffn(norm2(h))
    CAIF_DeviceTensor norm2_out=_norm2->Forward(h,training);
    CAIF_DeviceTensor ffn_out=_ffn->Forward(norm2_out,training);
    CAIF_DeviceTensor output=CAIF_DeviceTensor::Uninitialized(h.Shape(),*_stream);
    CAIF_DeviceOps::Add(h,ffn_out,output);

    return output;
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceTransformerBlock::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("TransformerBlock: layer has been moved from");
    }

    // Backward through residual 2: out = h + ffn(norm2(h))
    CAIF_DeviceTensor d_norm2_out=_ffn->Backward(grad_output);
    CAIF_DeviceTensor d_h_from_ffn=_norm2->Backward(d_norm2_out);
    CAIF_DeviceTensor d_h=CAIF_DeviceTensor::Uninitialized(grad_output.Shape(),*_stream);
    CAIF_DeviceOps::Add(grad_output,d_h_from_ffn,d_h);

    // Backward through residual 1: h = x + attention(norm1(x))
    CAIF_DeviceTensor d_norm1_out=_attention->Backward(d_h);
    CAIF_DeviceTensor d_x_from_attn=_norm1->Backward(d_norm1_out);
    CAIF_DeviceTensor d_x=CAIF_DeviceTensor::Uninitialized(d_h.Shape(),*_stream);
    CAIF_DeviceOps::Add(d_h,d_x_from_attn,d_x);

    return d_x;
  }
  CAIF_CATCH_BLOCK()
}

void CAIF_DeviceTransformerBlock::ZeroGradients()
{
  try
  {
    _norm1->ZeroGradients();
    _attention->ZeroGradients();
    _norm2->ZeroGradients();
    _ffn->ZeroGradients();
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceTransformerBlock::ParameterTensorCount()const
{
  try
  {
    return _norm1->ParameterTensorCount()+
           _attention->ParameterTensorCount()+
           _norm2->ParameterTensorCount()+
           _ffn->ParameterTensorCount();
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTransformerBlock::SubLayerMapping_t
CAIF_DeviceTransformerBlock::MapIndex(size_t index)const
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

  const size_t ffn_count=_ffn->ParameterTensorCount();
  if(index<ffn_count)
  {
    mapping.sub_layer_idx=3;
    mapping.local_idx=index;
    return mapping;
  }

  THROW_CAIFE("TransformerBlock::MapIndex: index out of range");
}

CAIF_DeviceTensor &CAIF_DeviceTransformerBlock::ParameterTensor(size_t index)
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
    return _ffn->ParameterTensor(mapping.local_idx);
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceTransformerBlock::ParameterTensor(size_t index)const
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
    return _ffn->ParameterTensor(mapping.local_idx);
  }
  CAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceTransformerBlock::GradientTensor(size_t index)
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
    return _ffn->GradientTensor(mapping.local_idx);
  }
  CAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceTransformerBlock::GradientTensor(size_t index)const
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
    return _ffn->GradientTensor(mapping.local_idx);
  }
  CAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceTransformerBlock::TotalParameterCount()const
{
  try
  {
    return _norm1->TotalParameterCount()+
           _attention->TotalParameterCount()+
           _norm2->TotalParameterCount()+
           _ffn->TotalParameterCount();
  }
  CAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceTransformerBlock::Description()const
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
    return "TransformerBlock(dim="+std::to_string(_config.dim)+
           ",heads="+std::to_string(_config.num_heads)+
           ",kv_heads="+std::to_string(_config.num_kv_heads)+
           ",ffn_dim="+std::to_string(_effective_ffn_dim)+
           ",causal="+causal_str+")";
  }
  CAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceTransformerBlock::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;
    // Collect names from all sub-layers with appropriate prefixes
    auto norm1_names=_norm1->ParameterNames(prefix+"input_layernorm.");
    names.insert(names.end(),norm1_names.begin(),norm1_names.end());

    auto attn_names=_attention->ParameterNames(prefix+"self_attn.");
    names.insert(names.end(),attn_names.begin(),attn_names.end());

    auto norm2_names=_norm2->ParameterNames(prefix+"post_attention_layernorm.");
    names.insert(names.end(),norm2_names.begin(),norm2_names.end());

    auto ffn_names=_ffn->ParameterNames(prefix+"mlp.");
    names.insert(names.end(),ffn_names.begin(),ffn_names.end());

    return names;
  }
  CAIF_CATCH_BLOCK()
}

uint32_t CAIF_DeviceTransformerBlock::ComputeDefaultFFNDim(uint32_t dim)
{
  // LLaMA-style: round_to(4 * dim * 2/3, 256)
  uint32_t raw=g_caif_ffn_multiplier_numerator*
               dim*
               g_caif_ffn_gated_numerator/
               g_caif_ffn_gated_denominator;
  uint32_t aligned=((raw+g_caif_ffn_alignment-1)/g_caif_ffn_alignment)*
                   g_caif_ffn_alignment;
  return aligned;
}
