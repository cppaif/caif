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

#include "caif_device_transformer_model.h"
#include "caif_device_gated_activations.h"
#include "caif_exception.h"
#include "caif_constants.h"

namespace instance
{

CAIF_DeviceTransformerModel::CAIF_DeviceTransformerModel(const CAIF_DeviceTransformerModel::Config_t &config,
                                                       CAIF_CudaStream &stream):
                                                       CAIF_DeviceLayer(stream),
                                                       _config(config),
                                                       _embedding(),
                                                       _pos_enc(),
                                                       _blocks(),
                                                       _final_norm(),
                                                       _head(),
                                                       _param_offsets()
{
  try
  {
    // Validate config
    if(config.vocab_size==0)
    {
      THROW_CAIFE("DeviceTransformerModel: vocab_size must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("DeviceTransformerModel: dim must be > 0");
    }
    if(config.num_heads==0)
    {
      THROW_CAIFE("DeviceTransformerModel: num_heads must be > 0");
    }
    if(config.num_layers==0)
    {
      THROW_CAIFE("DeviceTransformerModel: num_layers must be > 0");
    }
    if(config.dim%config.num_heads!=0)
    {
      THROW_CAIFE("DeviceTransformerModel: dim must be divisible by num_heads");
    }

    // Default num_kv_heads to num_heads if not specified
    uint32_t num_kv_heads=config.num_kv_heads;
    if(num_kv_heads==0)
    {
      num_kv_heads=config.num_heads;
    }

    // Default output_dim to vocab_size if not specified
    uint32_t output_dim=config.output_dim;
    if(output_dim==0)
    {
      output_dim=config.vocab_size;
    }

    // 1. Create token embedding
    CAIF_DeviceTokenEmbedding::Config_t emb_config{config.vocab_size,config.dim};
    _embedding=std::make_unique<CAIF_DeviceTokenEmbedding>(emb_config,stream);

    // 2. Create positional encoding (if not using RoPE)
    if(config.use_rope==false)
    {
      CAIF_DevicePositionalEncoding::Config_t pe_config{config.max_seq_len,config.dim,config.pe_mode};
      _pos_enc=std::make_unique<CAIF_DevicePositionalEncoding>(pe_config,stream);
    }

    // 3. Create transformer blocks
    CAIF_DeviceTransformerBlock::TransformerBlockConfig_t block_config;
    block_config.dim=config.dim;
    block_config.num_heads=config.num_heads;
    block_config.num_kv_heads=num_kv_heads;
    block_config.ffn_dim=config.ffn_dim;
    block_config.dropout_rate=0.0f;
    block_config.causal=config.causal;
    block_config.use_rope=config.use_rope;
    block_config.rope_base=g_caif_rope_default_base;

    _blocks.reserve(config.num_layers);
    for(uint32_t i=0;i<config.num_layers;++i)
    {
      _blocks.push_back(std::make_unique<CAIF_DeviceTransformerBlock>(block_config,stream));
    }

    // 4. Create final RMSNorm
    _final_norm=std::make_unique<CAIF_DeviceRMSNorm>(config.dim,stream);

    // 5. Create output head
    CAIF_DeviceLinearHead::Config_t head_config{config.dim,output_dim,false};
    if(config.tie_weights==true)
    {
      // Tied to embedding table
      _head=std::make_unique<CAIF_DeviceLinearHead>(head_config,
                                                   _embedding->ParameterTensor(0),
                                                   _embedding->GradientTensor(0),
                                                   stream);
    }
    else
    {
      _head=std::make_unique<CAIF_DeviceLinearHead>(head_config,stream);
    }

    // Build parameter offset table for MapIndex
    _param_offsets.clear();
    size_t offset=0;

    // Embedding
    _param_offsets.push_back(offset);
    offset+=_embedding->ParameterTensorCount();

    // Positional encoding (if present)
    if(_pos_enc!=nullptr)
    {
      _param_offsets.push_back(offset);
      offset+=_pos_enc->ParameterTensorCount();
    }

    // Transformer blocks
    for(size_t i=0;i<_blocks.size();++i)
    {
      _param_offsets.push_back(offset);
      offset+=_blocks[i]->ParameterTensorCount();
    }

    // Final norm
    _param_offsets.push_back(offset);
    offset+=_final_norm->ParameterTensorCount();

    // Head
    _param_offsets.push_back(offset);
    offset+=_head->ParameterTensorCount();

    // Total (sentinel)
    _param_offsets.push_back(offset);
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTransformerModel::CAIF_DeviceTransformerModel(CAIF_DeviceTransformerModel &&other)noexcept:
                                                       CAIF_DeviceLayer(std::move(other)),
                                                       _config(other._config),
                                                       _embedding(std::move(other._embedding)),
                                                       _pos_enc(std::move(other._pos_enc)),
                                                       _blocks(std::move(other._blocks)),
                                                       _final_norm(std::move(other._final_norm)),
                                                       _head(std::move(other._head)),
                                                       _param_offsets(std::move(other._param_offsets))
{
}

CAIF_DeviceTransformerModel &CAIF_DeviceTransformerModel::operator=(CAIF_DeviceTransformerModel &&other)noexcept
{
  if(this!=&other)
  {
    CAIF_DeviceLayer::operator=(std::move(other));
    _config=other._config;
    _embedding=std::move(other._embedding);
    _pos_enc=std::move(other._pos_enc);
    _blocks=std::move(other._blocks);
    _final_norm=std::move(other._final_norm);
    _head=std::move(other._head);
    _param_offsets=std::move(other._param_offsets);
  }
  return *this;
}

CAIF_DeviceTensor CAIF_DeviceTransformerModel::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceTransformerModel: model has been moved from");
    }

    // Input should be [batch, seq_len] with token IDs as float
    const auto &shape=input.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("DeviceTransformerModel::Forward: input must be 2D [batch, seq_len]");
    }

    // 1. Token embedding: [B, S] -> [B, S, dim]
    CAIF_DeviceTensor x=_embedding->Forward(input,training);

    // 2. Positional encoding (if not using RoPE)
    if(_pos_enc!=nullptr)
    {
      x=_pos_enc->Forward(x,training);
    }

    // 3. Transformer blocks
    for(size_t i=0;i<_blocks.size();++i)
    {
      x=_blocks[i]->Forward(x,training);
    }

    // 4. Final RMSNorm
    x=_final_norm->Forward(x,training);

    // 5. Output head: [B, S, dim] -> [B, S, output_dim]
    x=_head->Forward(x,training);

    return x;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceTransformerModel::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("DeviceTransformerModel: model has been moved from");
    }

    // Backward through head
    CAIF_DeviceTensor grad=_head->Backward(grad_output);

    // Backward through final norm
    grad=_final_norm->Backward(grad);

    // Backward through transformer blocks (reverse order)
    for(int i=static_cast<int>(_blocks.size())-1;i>=0;--i)
    {
      grad=_blocks[static_cast<size_t>(i)]->Backward(grad);
    }

    // Backward through positional encoding
    if(_pos_enc!=nullptr)
    {
      grad=_pos_enc->Backward(grad);
    }

    // Backward through embedding (returns empty tensor for token embedding)
    grad=_embedding->Backward(grad);

    return grad;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceTransformerModel::ZeroGradients()
{
  try
  {
    _embedding->ZeroGradients();

    if(_pos_enc!=nullptr)
    {
      _pos_enc->ZeroGradients();
    }

    for(size_t i=0;i<_blocks.size();++i)
    {
      _blocks[i]->ZeroGradients();
    }

    _final_norm->ZeroGradients();
    _head->ZeroGradients();
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceTransformerModel::MapIndex(size_t global_index,
                                          size_t &component_idx,
                                          size_t &local_idx)const
{
  // Find which component owns this index using binary search-like approach
  // _param_offsets has N+1 entries where the last is the total count

  for(size_t i=0;i<_param_offsets.size()-1;++i)
  {
    if(global_index<_param_offsets[i+1])
    {
      component_idx=i;
      local_idx=global_index-_param_offsets[i];
      return;
    }
  }

  THROW_CAIFE("DeviceTransformerModel::MapIndex: index out of range");
}

size_t CAIF_DeviceTransformerModel::ParameterTensorCount()const
{
  try
  {
    if(_param_offsets.empty()==true)
    {
      return 0;
    }
    return _param_offsets.back();
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceTransformerModel::ParameterTensor(size_t index)
{
  try
  {
    size_t comp_idx=0;
    size_t local_idx=0;
    MapIndex(index,comp_idx,local_idx);

    // Determine which component based on comp_idx
    size_t current_comp=0;

    // Embedding
    if(comp_idx==current_comp)
    {
      return _embedding->ParameterTensor(local_idx);
    }
    ++current_comp;

    // Positional encoding
    if(_pos_enc!=nullptr)
    {
      if(comp_idx==current_comp)
      {
        return _pos_enc->ParameterTensor(local_idx);
      }
      ++current_comp;
    }

    // Transformer blocks
    for(size_t i=0;i<_blocks.size();++i)
    {
      if(comp_idx==current_comp)
      {
        return _blocks[i]->ParameterTensor(local_idx);
      }
      ++current_comp;
    }

    // Final norm
    if(comp_idx==current_comp)
    {
      return _final_norm->ParameterTensor(local_idx);
    }
    ++current_comp;

    // Head
    if(comp_idx==current_comp)
    {
      return _head->ParameterTensor(local_idx);
    }

    THROW_CAIFE("DeviceTransformerModel::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceTransformerModel::ParameterTensor(size_t index)const
{
  try
  {
    size_t comp_idx=0;
    size_t local_idx=0;
    MapIndex(index,comp_idx,local_idx);

    size_t current_comp=0;

    if(comp_idx==current_comp)
    {
      return _embedding->ParameterTensor(local_idx);
    }
    ++current_comp;

    if(_pos_enc!=nullptr)
    {
      if(comp_idx==current_comp)
      {
        return _pos_enc->ParameterTensor(local_idx);
      }
      ++current_comp;
    }

    for(size_t i=0;i<_blocks.size();++i)
    {
      if(comp_idx==current_comp)
      {
        return _blocks[i]->ParameterTensor(local_idx);
      }
      ++current_comp;
    }

    if(comp_idx==current_comp)
    {
      return _final_norm->ParameterTensor(local_idx);
    }
    ++current_comp;

    if(comp_idx==current_comp)
    {
      return _head->ParameterTensor(local_idx);
    }

    THROW_CAIFE("DeviceTransformerModel::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceTransformerModel::GradientTensor(size_t index)
{
  try
  {
    size_t comp_idx=0;
    size_t local_idx=0;
    MapIndex(index,comp_idx,local_idx);

    size_t current_comp=0;

    if(comp_idx==current_comp)
    {
      return _embedding->GradientTensor(local_idx);
    }
    ++current_comp;

    if(_pos_enc!=nullptr)
    {
      if(comp_idx==current_comp)
      {
        return _pos_enc->GradientTensor(local_idx);
      }
      ++current_comp;
    }

    for(size_t i=0;i<_blocks.size();++i)
    {
      if(comp_idx==current_comp)
      {
        return _blocks[i]->GradientTensor(local_idx);
      }
      ++current_comp;
    }

    if(comp_idx==current_comp)
    {
      return _final_norm->GradientTensor(local_idx);
    }
    ++current_comp;

    if(comp_idx==current_comp)
    {
      return _head->GradientTensor(local_idx);
    }

    THROW_CAIFE("DeviceTransformerModel::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceTransformerModel::GradientTensor(size_t index)const
{
  try
  {
    size_t comp_idx=0;
    size_t local_idx=0;
    MapIndex(index,comp_idx,local_idx);

    size_t current_comp=0;

    if(comp_idx==current_comp)
    {
      return _embedding->GradientTensor(local_idx);
    }
    ++current_comp;

    if(_pos_enc!=nullptr)
    {
      if(comp_idx==current_comp)
      {
        return _pos_enc->GradientTensor(local_idx);
      }
      ++current_comp;
    }

    for(size_t i=0;i<_blocks.size();++i)
    {
      if(comp_idx==current_comp)
      {
        return _blocks[i]->GradientTensor(local_idx);
      }
      ++current_comp;
    }

    if(comp_idx==current_comp)
    {
      return _final_norm->GradientTensor(local_idx);
    }
    ++current_comp;

    if(comp_idx==current_comp)
    {
      return _head->GradientTensor(local_idx);
    }

    THROW_CAIFE("DeviceTransformerModel::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceTransformerModel::TotalParameterCount()const
{
  try
  {
    size_t total=_embedding->TotalParameterCount();

    if(_pos_enc!=nullptr)
    {
      total+=_pos_enc->TotalParameterCount();
    }

    for(size_t i=0;i<_blocks.size();++i)
    {
      total+=_blocks[i]->TotalParameterCount();
    }

    total+=_final_norm->TotalParameterCount();
    total+=_head->TotalParameterCount();

    return total;
  }
  CCAIF_CATCH_BLOCK()
}

std::string CAIF_DeviceTransformerModel::Description()const
{
  try
  {
    return "TransformerModel(dim="+std::to_string(_config.dim)+
           ",heads="+std::to_string(_config.num_heads)+
           ",layers="+std::to_string(_config.num_layers)+
           ",vocab="+std::to_string(_config.vocab_size)+")";
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceTransformerModel::ParameterNames(const std::string &prefix)const
{
  try
  {
    std::vector<std::string> names;

    // Embedding
    auto emb_names=_embedding->ParameterNames(prefix+"embed_tokens.");
    names.insert(names.end(),emb_names.begin(),emb_names.end());

    // Positional encoding (if present)
    if(_pos_enc!=nullptr)
    {
      auto pe_names=_pos_enc->ParameterNames(prefix+"embed_positions.");
      names.insert(names.end(),pe_names.begin(),pe_names.end());
    }

    // Transformer blocks
    for(size_t i=0;i<_blocks.size();++i)
    {
      std::string block_prefix=prefix+"layers."+std::to_string(i)+".";
      auto block_names=_blocks[i]->ParameterNames(block_prefix);
      names.insert(names.end(),block_names.begin(),block_names.end());
    }

    // Final norm
    auto norm_names=_final_norm->ParameterNames(prefix+"norm.");
    names.insert(names.end(),norm_names.begin(),norm_names.end());

    // Head
    auto head_names=_head->ParameterNames(prefix+"lm_head.");
    names.insert(names.end(),head_names.begin(),head_names.end());

    return names;
  }
  CCAIF_CATCH_BLOCK()
}

}//end instance namespace
