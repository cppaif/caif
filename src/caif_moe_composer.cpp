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

#include "caif_moe_composer.h"
#include "caif_device_rmsnorm.h"
#include "caif_device_rmsnorm_factory.h"
#include "caif_device_multi_head_attention.h"
#include "caif_device_token_embedding.h"
#include "caif_device_linear_head.h"
#include "caif_device_layer_typed.h"
#include "caif_exception.h"

namespace instance
{

template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DevicePreNormBlock<ComputeT,StorageT>> CAIF_MoEComposer::BuildMoEBlockImpl(
                                                                           const CAIF_MoEComposerBlockConfig &cfg,
                                                                           CAIF_CudaStream &stream)
{
  try
  {
    if(cfg.Dim()==0)
    {
      THROW_CAIFE("MoEComposer: block dim must be > 0");
    }
    if(cfg.NumHeads()==0)
    {
      THROW_CAIFE("MoEComposer: num_heads must be > 0");
    }
    if(cfg.NumKvHeads()==0)
    {
      THROW_CAIFE("MoEComposer: num_kv_heads must be > 0");
    }
    if(cfg.Dim()%cfg.NumHeads()!=0)
    {
      THROW_CAIFE("MoEComposer: dim must be divisible by num_heads");
    }
    if(cfg.MoeNumExperts()==0)
    {
      THROW_CAIFE("MoEComposer: moe_num_experts must be > 0");
    }
    if(cfg.MoeTopK()==0||cfg.MoeTopK()>cfg.MoeNumExperts())
    {
      THROW_CAIFE("MoEComposer: moe_top_k must be > 0 and <= moe_num_experts");
    }
    if(cfg.MoeInputDim()!=cfg.Dim())
    {
      THROW_CAIFE("MoEComposer: moe_input_dim must equal block dim");
    }

    auto norm1=CAIF_DeviceRMSNormFactory::Create(cfg.Dim(),
                                                 stream,
                                                 CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype(),
                                                 CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype(),
                                                 cfg.NormEps());

    CAIF_DeviceMultiHeadAttentionConfig attn_cfg(cfg.Dim(),
                                                 cfg.NumHeads(),
                                                 cfg.NumKvHeads(),
                                                 cfg.Dim()/cfg.NumHeads(),
                                                 cfg.Causal(),
                                                 cfg.UseRope(),
                                                 cfg.RopeBase(),
                                                 cfg.AttentionDropout());
    attn_cfg.SetRopeStyle(cfg.RopeStyle());
    auto attn=std::make_unique<CAIF_DeviceMultiHeadAttention<ComputeT,StorageT>>(attn_cfg,stream);

    auto norm2=CAIF_DeviceRMSNormFactory::Create(cfg.Dim(),
                                                 stream,
                                                 CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype(),
                                                 CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype(),
                                                 cfg.NormEps());

    auto moe=std::make_unique<CAIF_DeviceMoELayer<ComputeT,StorageT>>(cfg.MoeInputDim(),
                                                                      cfg.MoeHiddenDim(),
                                                                      cfg.MoeNumExperts(),
                                                                      cfg.MoeTopK(),
                                                                      cfg.MoeExpertUseGated(),
                                                                      cfg.MoeExpertUseBias(),
                                                                      cfg.MoeNumSharedExperts(),
                                                                      cfg.MoeSharedHiddenDim(),
                                                                      cfg.MoeRouterUseBias(),
                                                                      cfg.MoeRouterNoiseStd(),
                                                                      cfg.MoeCapacityFactor(),
                                                                      cfg.MoeOverflowStrategy(),
                                                                      cfg.MoeBalanceLossWeight(),
                                                                      cfg.MoeZLossWeight(),
                                                                      stream);

    typename CAIF_DevicePreNormBlock<ComputeT,StorageT>::SubLayerVec_t sublayers;
    sublayers.reserve(2);

    typename CAIF_DevicePreNormBlock<ComputeT,StorageT>::SubLayer_t sl_attn;
    sl_attn.norm_prefix =cfg.Norm1Prefix();
    sl_attn.layer_prefix=cfg.AttnPrefix();
    sl_attn.norm =std::move(norm1);
    sl_attn.layer=std::move(attn);
    sublayers.push_back(std::move(sl_attn));

    typename CAIF_DevicePreNormBlock<ComputeT,StorageT>::SubLayer_t sl_moe;
    sl_moe.norm_prefix =cfg.Norm2Prefix();
    sl_moe.layer_prefix=cfg.MoePrefix();
    sl_moe.norm =std::move(norm2);
    sl_moe.layer=std::move(moe);
    sublayers.push_back(std::move(sl_moe));

    return std::make_unique<CAIF_DevicePreNormBlock<ComputeT,StorageT>>(std::move(sublayers),stream);
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

std::unique_ptr<CAIF_DevicePreNormBlock<float,float>> CAIF_MoEComposer::BuildMoEBlock(
                                                                           const CAIF_MoEComposerBlockConfig &cfg,
                                                                           CAIF_CudaStream &stream)
{
  return BuildMoEBlockImpl<float,float>(cfg,stream);
}

template<typename ComputeT,typename StorageT>
std::unique_ptr<CAIF_DeviceNetwork> CAIF_MoEComposer::BuildModelImpl(const CAIF_MoEComposerModelConfig &cfg,
                                                                     CAIF_CudaStream &stream)
{
  try
  {
    if(cfg.VocabSize()==0)
    {
      THROW_CAIFE("MoEComposer: vocab_size must be > 0");
    }
    if(cfg.NumLayers()==0)
    {
      THROW_CAIFE("MoEComposer: num_layers must be > 0");
    }
    if(cfg.BlockTemplate().Dim()==0)
    {
      THROW_CAIFE("MoEComposer: block_template.dim must be > 0");
    }

    auto network=std::make_unique<CAIF_DeviceNetwork>(stream);

    const uint32_t dim=cfg.BlockTemplate().Dim();
    uint32_t output_dim=cfg.OutputDim();
    if(output_dim==0) output_dim=cfg.VocabSize();

    // Embedding / positional / head are built at <ComputeT, StorageT> so the
    // whole model (including a bf16 DSv2-Lite) assembles at the config dtype;
    // the public BuildModel below dispatches to the concrete instantiation.
    CAIF_DeviceTokenEmbeddingConfig emb_cfg{cfg.VocabSize(),dim};
    auto embedding=std::make_unique<CAIF_DeviceTokenEmbedding<ComputeT,StorageT>>(emb_cfg,stream);
    CAIF_DeviceTokenEmbedding<ComputeT,StorageT> *embedding_raw=embedding.get();
    network->AddLayer(std::move(embedding));

    if(cfg.BlockTemplate().UseRope()==false&&
       cfg.PeMode()!=CAIF_PositionalEncodingMode::CAIF_PositionalEncodingMode_e::None)
    {
      CAIF_DevicePositionalEncodingConfig pe_cfg{cfg.MaxSeqLen(),dim,cfg.PeMode()};
      network->AddLayer(std::make_unique<CAIF_DevicePositionalEncoding<ComputeT,StorageT>>(pe_cfg,
                                                                                           stream));
    }

    CAIF_MoEComposerBlockConfig block_cfg=cfg.BlockTemplate();
    for(uint32_t i=0;i<cfg.NumLayers();++i)
    {
      network->AddLayer(BuildMoEBlockImpl<ComputeT,StorageT>(block_cfg,stream));
    }

    network->AddLayer(CAIF_DeviceRMSNormFactory::Create(dim,
                                                        stream,
                                                        CAIF_DeviceLayerTyped<ComputeT,StorageT>::ComputeDtype(),
                                                        CAIF_DeviceLayerTyped<ComputeT,StorageT>::StorageDtype(),
                                                        cfg.FinalNormEps()));

    CAIF_DeviceLinearHeadConfig head_cfg{dim,output_dim,false};
    if(cfg.TieWeights()==true)
    {
      network->AddLayer(std::make_unique<CAIF_DeviceLinearHead<ComputeT,StorageT>>(
                                                                      head_cfg,
                                                                      embedding_raw->ParameterTensor(0),
                                                                      embedding_raw->GradientTensor(0),
                                                                      stream));
    }
    else
    {
      network->AddLayer(std::make_unique<CAIF_DeviceLinearHead<ComputeT,StorageT>>(head_cfg,stream));
    }

    return network;
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

std::unique_ptr<CAIF_DeviceNetwork> CAIF_MoEComposer::BuildModel(const CAIF_MoEComposerModelConfig &cfg,
                                                                 CAIF_CudaStream &stream)
{
  try
  {
    const CAIF_DataType::CAIF_DataType_e cd=cfg.ComputeDtype();
    const CAIF_DataType::CAIF_DataType_e sd=cfg.StorageDtype();
    if(cd==CAIF_DataType::CAIF_DataType_e::Float32&&sd==CAIF_DataType::CAIF_DataType_e::Float32)
    {
      return BuildModelImpl<float,float>(cfg,stream);
    }
#ifdef USE_CAIF_CUDA
    if(cd==CAIF_DataType::CAIF_DataType_e::Float32&&sd==CAIF_DataType::CAIF_DataType_e::Float16)
    {
      return BuildModelImpl<float,__half>(cfg,stream);
    }
    if(cd==CAIF_DataType::CAIF_DataType_e::Float32&&sd==CAIF_DataType::CAIF_DataType_e::BFloat16)
    {
      return BuildModelImpl<float,__nv_bfloat16>(cfg,stream);
    }
#endif
    THROW_CAIFE("MoEComposer: BuildModel supports compute=fp32 with storage in {fp32,fp16,bf16}");
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
