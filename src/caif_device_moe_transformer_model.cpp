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
// Device-resident MoE Transformer Model implementation
//------------------------------------------------------------------------------
#include "caif_device_moe_transformer_model.h"
#include "caif_device_gated_activations.h"
#include "caif_exception.h"
#include "caif_constants.h"

using namespace instance;

CAIF_DeviceMoETransformerModel::CAIF_DeviceMoETransformerModel(
  const Config_t &config,
  CAIF_CudaStream &stream):CAIF_DeviceLayer(stream),
                           _config(config),
                           _num_moe_layers(0),
                           _embedding(),
                           _pos_enc(),
                           _blocks(),
                           _block_is_moe(),
                           _final_norm(),
                           _head(),
                           _param_offsets(),
                           _total_aux_losses({0.0f,0.0f})
{
  try
  {
    // Validate config
    if(config.vocab_size==0)
    {
      THROW_CAIFE("MoETransformerModel: vocab_size must be > 0");
    }
    if(config.dim==0)
    {
      THROW_CAIFE("MoETransformerModel: dim must be > 0");
    }
    if(config.num_heads==0)
    {
      THROW_CAIFE("MoETransformerModel: num_heads must be > 0");
    }
    if(config.num_layers==0)
    {
      THROW_CAIFE("MoETransformerModel: num_layers must be > 0");
    }
    if(config.dim%config.num_heads!=0)
    {
      THROW_CAIFE("MoETransformerModel: dim must be divisible by num_heads");
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

    // 3. Create transformer blocks (mixed dense/MoE)
    _blocks.reserve(config.num_layers);
    _block_is_moe.reserve(config.num_layers);

    for(uint32_t i=0;i<config.num_layers;++i)
    {
      bool is_moe=IsMoELayer(i);
      _block_is_moe.push_back(is_moe);

      if(is_moe==true)
      {
        // Create MoE block
        CAIF_DeviceMoEBlock::Config_t moe_config;
        moe_config.dim=config.dim;
        moe_config.num_heads=config.num_heads;
        moe_config.num_kv_heads=num_kv_heads;
        moe_config.dropout_rate=0.0f;
        moe_config.causal=config.causal;
        moe_config.use_rope=config.use_rope;
        moe_config.rope_base=config.rope_base;
        moe_config.num_experts=config.num_experts;
        moe_config.top_k=config.top_k;
        moe_config.expert_ffn_dim=config.expert_ffn_dim;
        moe_config.expert_use_gated=config.expert_use_gated;
        moe_config.num_shared_experts=config.num_shared_experts;
        moe_config.shared_ffn_dim=config.shared_ffn_dim;
        moe_config.capacity_factor=config.capacity_factor;
        moe_config.overflow_strategy=config.overflow_strategy;
        moe_config.balance_loss_weight=config.balance_loss_weight;
        moe_config.z_loss_weight=config.z_loss_weight;
        moe_config.router_noise_std=config.router_noise_std;

        _blocks.push_back(std::make_unique<CAIF_DeviceMoEBlock>(moe_config,stream));
        ++_num_moe_layers;
      }
      else
      {
        // Create dense block
        CAIF_DeviceTransformerBlock::TransformerBlockConfig_t block_config;
        block_config.dim=config.dim;
        block_config.num_heads=config.num_heads;
        block_config.num_kv_heads=num_kv_heads;
        block_config.ffn_dim=config.ffn_dim;
        block_config.dropout_rate=0.0f;
        block_config.causal=config.causal;
        block_config.use_rope=config.use_rope;
        block_config.rope_base=config.rope_base;

        _blocks.push_back(std::make_unique<CAIF_DeviceTransformerBlock>(block_config,stream));
      }
    }

    // 4. Create final RMSNorm
    _final_norm=std::make_unique<CAIF_DeviceRMSNorm>(config.dim,stream);

    // 5. Create output head
    CAIF_DeviceLinearHead::Config_t head_config{config.dim,output_dim,false};
    if(config.tie_weights==true)
    {
      _head=std::make_unique<CAIF_DeviceLinearHead>(head_config,
                                                   _embedding->ParameterTensor(0),
                                                   _embedding->GradientTensor(0),
                                                   stream);
    }
    else
    {
      _head=std::make_unique<CAIF_DeviceLinearHead>(head_config,stream);
    }

    // Build parameter offset table
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

    // Blocks
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

CAIF_DeviceMoETransformerModel::CAIF_DeviceMoETransformerModel(
  CAIF_DeviceMoETransformerModel &&other):CAIF_DeviceLayer(std::move(other)),
                                         _config(other._config),
                                         _num_moe_layers(other._num_moe_layers),
                                         _embedding(std::move(other._embedding)),
                                         _pos_enc(std::move(other._pos_enc)),
                                         _blocks(std::move(other._blocks)),
                                         _block_is_moe(std::move(other._block_is_moe)),
                                         _final_norm(std::move(other._final_norm)),
                                         _head(std::move(other._head)),
                                         _param_offsets(std::move(other._param_offsets)),
                                         _total_aux_losses(other._total_aux_losses)
{
}

CAIF_DeviceMoETransformerModel &CAIF_DeviceMoETransformerModel::operator=(CAIF_DeviceMoETransformerModel &&other)
{
  try
  {
    if(this!=&other)
    {
      CAIF_DeviceLayer::operator=(std::move(other));
      _config=other._config;
      _num_moe_layers=other._num_moe_layers;
      _embedding=std::move(other._embedding);
      _pos_enc=std::move(other._pos_enc);
      _blocks=std::move(other._blocks);
      _block_is_moe=std::move(other._block_is_moe);
      _final_norm=std::move(other._final_norm);
      _head=std::move(other._head);
      _param_offsets=std::move(other._param_offsets);
      _total_aux_losses=other._total_aux_losses;
    }
    return *this;
  }
  CCAIF_CATCH_BLOCK()
}

bool CAIF_DeviceMoETransformerModel::IsMoELayer(uint32_t layer_idx)const
{
  // moe_layer_interval=0: All dense (no MoE)
  // moe_layer_interval=1: All MoE
  // moe_layer_interval=2: Every 2nd layer (1, 3, 5, ...) is MoE
  // moe_layer_interval=N: Every Nth layer (N-1, 2N-1, 3N-1, ...) is MoE

  if(_config.moe_layer_interval==0)
  {
    return false;
  }
  if(_config.moe_layer_interval==1)
  {
    return true;
  }

  // For interval > 1: layer is MoE if (layer_idx + 1) % interval == 0
  return ((layer_idx+1)%_config.moe_layer_interval)==0;
}

CAIF_DeviceTensor CAIF_DeviceMoETransformerModel::Forward(const CAIF_DeviceTensor &input,bool training)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("MoETransformerModel: model has been moved from");
    }

    const auto &shape=input.Shape();
    if(shape.size()!=2)
    {
      THROW_CAIFE("MoETransformerModel::Forward: input must be 2D [batch, seq_len]");
    }

    // Reset auxiliary losses
    _total_aux_losses.balance_loss=0.0f;
    _total_aux_losses.z_loss=0.0f;

    // 1. Token embedding
    CAIF_DeviceTensor x=_embedding->Forward(input,training);

    // 2. Positional encoding (if not using RoPE)
    if(_pos_enc!=nullptr)
    {
      x=_pos_enc->Forward(x,training);
    }

    // 3. Transformer/MoE blocks
    for(size_t i=0;i<_blocks.size();++i)
    {
      x=_blocks[i]->Forward(x,training);

      // Accumulate aux losses from MoE blocks
      if(_block_is_moe[i]==true&&training==true)
      {
        CAIF_DeviceMoEBlock *moe_block=static_cast<CAIF_DeviceMoEBlock*>(_blocks[i].get());
        const auto &aux=moe_block->LastAuxLosses();
        _total_aux_losses.balance_loss+=aux.balance_loss;
        _total_aux_losses.z_loss+=aux.z_loss;
      }
    }

    // 4. Final RMSNorm
    x=_final_norm->Forward(x,training);

    // 5. Output head
    x=_head->Forward(x,training);

    return x;
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor CAIF_DeviceMoETransformerModel::Backward(const CAIF_DeviceTensor &grad_output)
{
  try
  {
    if(_stream==nullptr)
    {
      THROW_CAIFE("MoETransformerModel: model has been moved from");
    }

    // Backward through head
    CAIF_DeviceTensor grad=_head->Backward(grad_output);

    // Backward through final norm
    grad=_final_norm->Backward(grad);

    // Backward through blocks (reverse order)
    for(int i=static_cast<int>(_blocks.size())-1;i>=0;--i)
    {
      grad=_blocks[static_cast<size_t>(i)]->Backward(grad);
    }

    // Backward through positional encoding
    if(_pos_enc!=nullptr)
    {
      grad=_pos_enc->Backward(grad);
    }

    // Backward through embedding
    grad=_embedding->Backward(grad);

    return grad;
  }
  CCAIF_CATCH_BLOCK()
}

void CAIF_DeviceMoETransformerModel::ZeroGradients()
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

void CAIF_DeviceMoETransformerModel::MapIndex(size_t global_index,
                                              size_t &component_idx,
                                              size_t &local_idx)const
{
  for(size_t i=0;i<_param_offsets.size()-1;++i)
  {
    if(global_index<_param_offsets[i+1])
    {
      component_idx=i;
      local_idx=global_index-_param_offsets[i];
      return;
    }
  }

  THROW_CAIFE("MoETransformerModel::MapIndex: index out of range");
}

size_t CAIF_DeviceMoETransformerModel::ParameterTensorCount()const
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

CAIF_DeviceTensor &CAIF_DeviceMoETransformerModel::ParameterTensor(size_t index)
{
  try
  {
    size_t comp_idx=0;
    size_t local_idx=0;
    MapIndex(index,comp_idx,local_idx);

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

    // Blocks
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

    THROW_CAIFE("MoETransformerModel::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceMoETransformerModel::ParameterTensor(size_t index)const
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

    THROW_CAIFE("MoETransformerModel::ParameterTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

CAIF_DeviceTensor &CAIF_DeviceMoETransformerModel::GradientTensor(size_t index)
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

    THROW_CAIFE("MoETransformerModel::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

const CAIF_DeviceTensor &CAIF_DeviceMoETransformerModel::GradientTensor(size_t index)const
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

    THROW_CAIFE("MoETransformerModel::GradientTensor: index out of range");
  }
  CCAIF_CATCH_BLOCK()
}

size_t CAIF_DeviceMoETransformerModel::TotalParameterCount()const
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

std::string CAIF_DeviceMoETransformerModel::Description()const
{
  try
  {
    return "MoETransformerModel(dim="+std::to_string(_config.dim)+
           ",heads="+std::to_string(_config.num_heads)+
           ",layers="+std::to_string(_config.num_layers)+
           ",moe_layers="+std::to_string(_num_moe_layers)+
           ",experts="+std::to_string(_config.num_experts)+
           ",vocab="+std::to_string(_config.vocab_size)+")";
  }
  CCAIF_CATCH_BLOCK()
}

std::vector<std::string> CAIF_DeviceMoETransformerModel::ParameterNames(const std::string &prefix)const
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

    // Blocks
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
