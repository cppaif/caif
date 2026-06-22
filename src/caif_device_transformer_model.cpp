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
#include "caif_serialization_constants.h"
#include "caif_role_registry.h"

namespace instance
{

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerModel<ComputeT,StorageT>::CAIF_DeviceTransformerModel(
                                                  const CAIF_DeviceTransformerModelConfig &config,
                                                  CAIF_CudaStream &stream):CAIF_DeviceContainer(stream),
                                                                           _config(config),
                                                                           _pos_enc_present(false)
{
  try
  {
    // Validate config
    if(config.VocabSize()==0)
    {
      THROW_CAIFE("DeviceTransformerModel: vocab_size must be > 0");
    }
    if(config.Dim()==0)
    {
      THROW_CAIFE("DeviceTransformerModel: dim must be > 0");
    }
    if(config.NumHeads()==0)
    {
      THROW_CAIFE("DeviceTransformerModel: num_heads must be > 0");
    }
    if(config.NumLayers()==0)
    {
      THROW_CAIFE("DeviceTransformerModel: num_layers must be > 0");
    }
    if(config.Dim()%config.NumHeads()!=0)
    {
      THROW_CAIFE("DeviceTransformerModel: dim must be divisible by num_heads");
    }

    // Default num_kv_heads to num_heads if not specified
    uint32_t num_kv_heads=config.NumKvHeads();
    if(num_kv_heads==0)
    {
      num_kv_heads=config.NumHeads();
    }

    // Default output_dim to vocab_size if not specified
    uint32_t output_dim=config.OutputDim();
    if(output_dim==0)
    {
      output_dim=config.VocabSize();
    }

    // 1. Token embedding — keep a temporary typed raw reference before
    //    transferring ownership to the container, because the tied-weights
    //    head needs the embedding's parameter+gradient tensors.
    CAIF_DeviceTokenEmbeddingConfig emb_config(config.VocabSize(),config.Dim());
    emb_config.SetOutputScale(config.EmbedScale());
    auto embedding=std::make_unique<CAIF_DeviceTokenEmbedding<ComputeT,StorageT>>(emb_config,stream);
    CAIF_DeviceTokenEmbedding<ComputeT,StorageT> *embedding_ref=embedding.get();
    AddLayer(std::move(embedding));

    // 2. Positional encoding (if not using RoPE)
    if(config.UseRope()==false)
    {
      CAIF_DevicePositionalEncodingConfig pe_config(config.MaxSeqLen(),config.Dim(),config.PeMode());
      AddLayer(std::make_unique<CAIF_DevicePositionalEncoding<ComputeT,StorageT>>(pe_config,stream));
      SetPosEncPresent(true);
    }

    // 3. Transformer blocks
    CAIF_DeviceTransformerBlockConfig block_config(config.Dim(),
                                                   config.NumHeads(),
                                                   num_kv_heads,
                                                   config.FfnDim(),
                                                   0.0f,
                                                   config.Causal(),
                                                   config.UseRope(),
                                                   g_caif_rope_default_base);
    block_config.SetRopeStyle(config.RopeStyle());

    for(uint32_t i=0;i<config.NumLayers();++i)
    {
      AddLayer(std::make_unique<CAIF_DeviceTransformerBlock<ComputeT,StorageT>>(block_config,stream));
    }

    // 4. Final RMSNorm
    AddLayer(std::make_unique<CAIF_DeviceRMSNorm<ComputeT,StorageT>>(config.Dim(),stream));

    // 5. Output head
    CAIF_DeviceLinearHeadConfig head_config(config.Dim(),output_dim,false);
    head_config.SetOutputScale(config.LogitScale());
    if(config.TieWeights()==true)
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
                                                                   _config(other.Config()),
                                                                   _pos_enc_present(other.PosEncPresent())
{
}

template<typename ComputeT,typename StorageT>
CAIF_DeviceTransformerModel<ComputeT,StorageT> &
CAIF_DeviceTransformerModel<ComputeT,StorageT>::operator=(CAIF_DeviceTransformerModel &&other)
{
  if(this!=&other)
  {
    CAIF_DeviceContainer::operator=(std::move(other));
    SetConfig(other.Config());
    SetPosEncPresent(other.PosEncPresent());
  }
  return *this;
}

template<typename ComputeT,typename StorageT>
std::string CAIF_DeviceTransformerModel<ComputeT,StorageT>::Description()const
{
  try
  {
    return std::string(g_serial_tag_transformer_model)+
           g_serial_open_paren+
           g_serial_kv_dim+
           std::to_string(Config().Dim())+
           g_serial_comma+
           g_serial_kv_heads+
           std::to_string(Config().NumHeads())+
           g_serial_comma+
           g_serial_kv_layers+
           std::to_string(Config().NumLayers())+
           g_serial_comma+
           g_serial_kv_vocab+
           std::to_string(Config().VocabSize())+
           g_serial_close_paren;
  }
  CAIF_CATCH_BLOCK()
}

template<typename ComputeT,typename StorageT>
std::vector<std::string>
CAIF_DeviceTransformerModel<ComputeT,StorageT>::ParameterNames(const std::string &prefix)const
{
  try
  {
    // Slot layout mirrors the order AddLayer was called in the ctor.
    // Both structural prefixes and sub-layer leaf names flow through
    // the registry — caller overrides any of them via SetName.
    const CAIF_RoleRegistry &reg=CAIF_RoleRegistry::Instance();
    std::vector<std::string> names;
    std::vector<std::string> slot_names;

    size_t slot=0;

    // Embedding
    slot_names=Layer(slot).ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::PathEmbedIn_e));
    names.insert(names.end(),slot_names.begin(),slot_names.end());
    ++slot;

    // Positional encoding (optional)
    if(PosEncPresent()==true)
    {
      slot_names=Layer(slot).ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::PathEmbedPos_e));
      names.insert(names.end(),slot_names.begin(),slot_names.end());
      ++slot;
    }

    // Transformer blocks
    for(uint32_t i=0;i<Config().NumLayers();++i)
    {
      std::string block_prefix=prefix+reg.Name(CAIF_ParamRole::Role_e::PathTransformerBlocks_e)+std::to_string(i)+".";
      slot_names=Layer(slot).ParameterNames(block_prefix);
      names.insert(names.end(),slot_names.begin(),slot_names.end());
      ++slot;
    }

    // Final norm
    slot_names=Layer(slot).ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::PathFinalNorm_e));
    names.insert(names.end(),slot_names.begin(),slot_names.end());
    ++slot;

    // Head
    slot_names=Layer(slot).ParameterNames(prefix+reg.Name(CAIF_ParamRole::Role_e::PathHead_e));
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
