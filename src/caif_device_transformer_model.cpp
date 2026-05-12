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
#include "caif_device_swiglu_activation.h"
#include "caif_device_geglu_activation.h"
#include "caif_device_reglu_activation.h"
#include "caif_device_glu_activation.h"
#include "caif_device_bilinear_activation.h"
#include "caif_exception.h"
#include "caif_constants.h"

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerModel<ComputeT,StorageT>::CAIF_DeviceTransformerModel(
                                                  const Config_t &config,
                                                  CAIF_CudaStream &stream):CAIF_DeviceContainer(stream),
                                                                           _config(config),
                                                                           _pos_enc_present(false)
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

    // 1. Token embedding — keep a temporary typed raw reference before
    //    transferring ownership to the container, because the tied-weights
    //    head needs the embedding's parameter+gradient tensors.
    typename CAIF_DeviceTokenEmbedding<ComputeT,StorageT>::Config_t emb_config{config.vocab_size,
                                                                                config.dim};
    auto embedding=std::make_unique<CAIF_DeviceTokenEmbedding<ComputeT,StorageT>>(emb_config,stream);
    CAIF_DeviceTokenEmbedding<ComputeT,StorageT> *embedding_ref=embedding.get();
    AddLayer(std::move(embedding));

    // 2. Positional encoding (if not using RoPE)
    if(config.use_rope==false)
    {
      typename CAIF_DevicePositionalEncoding<ComputeT,StorageT>::Config_t pe_config{config.max_seq_len,
                                                                                     config.dim,
                                                                                     config.pe_mode};
      AddLayer(std::make_unique<CAIF_DevicePositionalEncoding<ComputeT,StorageT>>(pe_config,stream));
      _pos_enc_present=true;
    }

    // 3. Transformer blocks
    typename CAIF_DeviceTransformerBlock<ComputeT,StorageT>::TransformerBlockConfig_t block_config;
    block_config.dim=config.dim;
    block_config.num_heads=config.num_heads;
    block_config.num_kv_heads=num_kv_heads;
    block_config.ffn_dim=config.ffn_dim;
    block_config.dropout_rate=0.0f;
    block_config.causal=config.causal;
    block_config.use_rope=config.use_rope;
    block_config.rope_base=g_caif_rope_default_base;
    block_config.rope_style=config.rope_style;

    for(uint32_t i=0;i<config.num_layers;++i)
    {
      AddLayer(std::make_unique<CAIF_DeviceTransformerBlock<ComputeT,StorageT>>(block_config,stream));
    }

    // 4. Final RMSNorm
    AddLayer(std::make_unique<CAIF_DeviceRMSNorm<ComputeT,StorageT>>(config.dim,stream));

    // 5. Output head
    typename CAIF_DeviceLinearHead<ComputeT,StorageT>::Config_t head_config{config.dim,
                                                                             output_dim,
                                                                             false};
    if(config.tie_weights==true)
    {
      AddLayer(std::make_unique<CAIF_DeviceLinearHead<ComputeT,StorageT>>(
                                                       head_config,
                                                       embedding_ref->ParameterTensor(0),
                                                       embedding_ref->GradientTensor(0),
                                                       stream));
    }
    else
    {
      AddLayer(std::make_unique<CAIF_DeviceLinearHead<ComputeT,StorageT>>(head_config,stream));
    }
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerModel<ComputeT,StorageT>::CAIF_DeviceTransformerModel(
                              CAIF_DeviceTransformerModel &&other):CAIF_DeviceContainer(std::move(other)),
                                                                   _config(other._config),
                                                                   _pos_enc_present(other._pos_enc_present)
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerModel<ComputeT,StorageT> &
CAIF_DeviceTransformerModel<ComputeT,StorageT>::operator=(CAIF_DeviceTransformerModel &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceContainer::operator=(std::move(other));
    _config=other._config;
    _pos_enc_present=other._pos_enc_present;
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceTransformerModel<ComputeT,StorageT>::Description()const
{
  try
  {
    return std::string(g_caif_description_transformer_model_prefix)+
           "(dim="+std::to_string(_config.dim)+
           ",heads="+std::to_string(_config.num_heads)+
           ",layers="+std::to_string(_config.num_layers)+
           ",vocab="+std::to_string(_config.vocab_size)+")";
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceTransformerModel<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    // Slot layout mirrors the order AddLayer was called in the ctor:
    //   [0]            embedding       -> "embed_tokens."
    //   [1]            pos_enc (opt)   -> "embed_positions."
    //   [1|2 .. n]     N blocks        -> "layers.N."
    //   [.]            final norm      -> "norm."
    //   [.]            head            -> "lm_head."
    std::vector<std::string> names;
    std::vector<std::string> slot_names;

    size_t slot=0;

    // Embedding
    slot_names=Layer(slot).ParameterNames(prefix+g_caif_name_embed_tokens);
    names.insert(names.end(),slot_names.begin(),slot_names.end());
    ++slot;

    // Positional encoding (optional)
    if(_pos_enc_present==true)
    {
      slot_names=Layer(slot).ParameterNames(prefix+g_caif_name_embed_positions);
      names.insert(names.end(),slot_names.begin(),slot_names.end());
      ++slot;
    }

    // Transformer blocks
    for(uint32_t i=0;i<_config.num_layers;++i)
    {
      std::string block_prefix=prefix+g_caif_name_layers_prefix+std::to_string(i)+".";
      slot_names=Layer(slot).ParameterNames(block_prefix);
      names.insert(names.end(),slot_names.begin(),slot_names.end());
      ++slot;
    }

    // Final norm
    slot_names=Layer(slot).ParameterNames(prefix+g_caif_name_norm);
    names.insert(names.end(),slot_names.begin(),slot_names.end());
    ++slot;

    // Head
    slot_names=Layer(slot).ParameterNames(prefix+g_caif_name_lm_head);
    names.insert(names.end(),slot_names.begin(),slot_names.end());

    return names;
  }
  CAIF_CATCH_BLOCK()
}

template class CAIF_DeviceTransformerModel<float,float>;
#ifdef USE_CAIF_CUDA
template class CAIF_DeviceTransformerModel<float,__half>;
template class CAIF_DeviceTransformerModel<float,__nv_bfloat16>;
template class CAIF_DeviceTransformerModel<__half,float>;
template class CAIF_DeviceTransformerModel<__half,__half>;
template class CAIF_DeviceTransformerModel<__half,__nv_bfloat16>;
template class CAIF_DeviceTransformerModel<__nv_bfloat16,float>;
template class CAIF_DeviceTransformerModel<__nv_bfloat16,__half>;
template class CAIF_DeviceTransformerModel<__nv_bfloat16,__nv_bfloat16>;
#endif

}//end instance namespace
