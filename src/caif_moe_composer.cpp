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
#include "caif_exception.h"

namespace instance
{

std::unique_ptr<CAIF_DevicePreNormBlock<float,float>>
  CAIF_MoEComposer::BuildMoEBlock(const BlockConfig_t &cfg,CAIF_CudaStream &stream)
{
  try
  {
    if(cfg.dim==0)
    {
      THROW_CAIFE("MoEComposer: block dim must be > 0");
    }
    if(cfg.num_heads==0)
    {
      THROW_CAIFE("MoEComposer: num_heads must be > 0");
    }
    if(cfg.num_kv_heads==0)
    {
      THROW_CAIFE("MoEComposer: num_kv_heads must be > 0");
    }
    if(cfg.dim%cfg.num_heads!=0)
    {
      THROW_CAIFE("MoEComposer: dim must be divisible by num_heads");
    }
    if(cfg.moe_num_experts==0)
    {
      THROW_CAIFE("MoEComposer: moe_num_experts must be > 0");
    }
    if(cfg.moe_top_k==0||cfg.moe_top_k>cfg.moe_num_experts)
    {
      THROW_CAIFE("MoEComposer: moe_top_k must be > 0 and <= moe_num_experts");
    }
    if(cfg.moe_input_dim!=cfg.dim)
    {
      THROW_CAIFE("MoEComposer: moe_input_dim must equal block dim");
    }

    auto norm1=CAIF_DeviceRMSNormFactory::Create(cfg.dim,
                                                 stream,
                                                 CAIF_DataType::CAIF_DataType_e::Float32,
                                                 CAIF_DataType::CAIF_DataType_e::Float32,
                                                 cfg.norm_eps);

    CAIF_DeviceMultiHeadAttention<float,float>::AttentionConfig_t attn_cfg;
    attn_cfg.dim          =cfg.dim;
    attn_cfg.num_heads    =cfg.num_heads;
    attn_cfg.num_kv_heads =cfg.num_kv_heads;
    attn_cfg.head_dim     =cfg.dim/cfg.num_heads;
    attn_cfg.causal       =cfg.causal;
    attn_cfg.use_rope     =cfg.use_rope;
    attn_cfg.rope_base    =cfg.rope_base;
    attn_cfg.rope_style   =cfg.rope_style;
    attn_cfg.dropout_rate =cfg.attention_dropout;
    auto attn=std::make_unique<CAIF_DeviceMultiHeadAttention<float,float>>(attn_cfg,stream);

    auto norm2=CAIF_DeviceRMSNormFactory::Create(cfg.dim,
                                                 stream,
                                                 CAIF_DataType::CAIF_DataType_e::Float32,
                                                 CAIF_DataType::CAIF_DataType_e::Float32,
                                                 cfg.norm_eps);

    auto moe=std::make_unique<CAIF_DeviceMoELayer<float,float>>(cfg.moe_input_dim,
                                                   cfg.moe_hidden_dim,
                                                   cfg.moe_num_experts,
                                                   cfg.moe_top_k,
                                                   cfg.moe_expert_use_gated,
                                                   cfg.moe_expert_use_bias,
                                                   cfg.moe_num_shared_experts,
                                                   cfg.moe_shared_hidden_dim,
                                                   cfg.moe_router_use_bias,
                                                   cfg.moe_router_noise_std,
                                                   cfg.moe_capacity_factor,
                                                   cfg.moe_overflow_strategy,
                                                   cfg.moe_balance_loss_weight,
                                                   cfg.moe_z_loss_weight,
                                                   stream);

    CAIF_DevicePreNormBlock<float,float>::SubLayerVec_t sublayers;
    sublayers.reserve(2);

    CAIF_DevicePreNormBlock<float,float>::SubLayer_t sl_attn;
    sl_attn.norm_prefix =cfg.norm1_prefix;
    sl_attn.layer_prefix=cfg.attn_prefix;
    sl_attn.norm =std::move(norm1);
    sl_attn.layer=std::move(attn);
    sublayers.push_back(std::move(sl_attn));

    CAIF_DevicePreNormBlock<float,float>::SubLayer_t sl_moe;
    sl_moe.norm_prefix =cfg.norm2_prefix;
    sl_moe.layer_prefix=cfg.moe_prefix;
    sl_moe.norm =std::move(norm2);
    sl_moe.layer=std::move(moe);
    sublayers.push_back(std::move(sl_moe));

    return std::make_unique<CAIF_DevicePreNormBlock<float,float>>(std::move(sublayers),stream);
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

std::unique_ptr<CAIF_DeviceNetwork>
  CAIF_MoEComposer::BuildModel(const ModelConfig_t &cfg,CAIF_CudaStream &stream)
{
  try
  {
    if(cfg.vocab_size==0)
    {
      THROW_CAIFE("MoEComposer: vocab_size must be > 0");
    }
    if(cfg.num_layers==0)
    {
      THROW_CAIFE("MoEComposer: num_layers must be > 0");
    }
    if(cfg.block_template.dim==0)
    {
      THROW_CAIFE("MoEComposer: block_template.dim must be > 0");
    }

    auto network=std::make_unique<CAIF_DeviceNetwork>(stream);

    const uint32_t dim=cfg.block_template.dim;
    uint32_t output_dim=cfg.output_dim;
    if(output_dim==0) output_dim=cfg.vocab_size;

    // TODO: paths in this composer still pin to <float, float> until
    // the embedding/head/positional layers all migrate. Replace with
    // the dtype-bearing factory once the rest of the model is templated.
    CAIF_DeviceTokenEmbedding<float,float>::Config_t emb_cfg{cfg.vocab_size,dim};
    auto embedding=std::make_unique<CAIF_DeviceTokenEmbedding<float,float>>(emb_cfg,stream);
    CAIF_DeviceTokenEmbedding<float,float> *embedding_raw=embedding.get();
    network->AddLayer(std::move(embedding));

    if(cfg.block_template.use_rope==false&&cfg.pe_mode!=PositionalEncodingMode_e::None)
    {
      CAIF_DevicePositionalEncoding<float,float>::Config_t pe_cfg{cfg.max_seq_len,
                                                                  dim,
                                                                  cfg.pe_mode};
      network->AddLayer(std::make_unique<CAIF_DevicePositionalEncoding<float,float>>(pe_cfg,
                                                                                      stream));
    }

    BlockConfig_t block_cfg=cfg.block_template;
    for(uint32_t i=0;i<cfg.num_layers;++i)
    {
      network->AddLayer(BuildMoEBlock(block_cfg,stream));
    }

    network->AddLayer(CAIF_DeviceRMSNormFactory::Create(dim,
                                                        stream,
                                                        CAIF_DataType::CAIF_DataType_e::Float32,
                                                        CAIF_DataType::CAIF_DataType_e::Float32,
                                                        cfg.final_norm_eps));

    CAIF_DeviceLinearHead<float,float>::Config_t head_cfg{dim,
                                             output_dim,
                                             false};
    if(cfg.tie_weights==true)
    {
      network->AddLayer(std::make_unique<CAIF_DeviceLinearHead<float,float>>(head_cfg,
                                                                embedding_raw->ParameterTensor(0),
                                                                embedding_raw->GradientTensor(0),
                                                                stream));
    }
    else
    {
      network->AddLayer(std::make_unique<CAIF_DeviceLinearHead<float,float>>(head_cfg,stream));
    }

    return network;
  }
  CAIF_CATCH_BLOCK()
  return nullptr;
}

}//end instance namespace
