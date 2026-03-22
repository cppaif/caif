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
// Retrainer - GLM-4.7-Flash Model Builder
//------------------------------------------------------------------------------
#include "rtnr_glm_model_builder.h"
#include "rtnr_constants.h"

#include "caif/caif_device_token_embedding.h"
#include "caif/caif_device_pre_norm_block.h"
#include "caif/caif_device_ml_attention.h"
#include "caif/caif_device_ffn.h"
#include "caif/caif_device_moe_layer.h"
#include "caif/caif_device_moe_expert.h"
#include "caif/caif_device_rmsnorm.h"
#include "caif/caif_device_linear_head.h"
#include "caif/caif_device_frozen_linear.h"
#include "caif/caif_device_ops.h"
#include "caif/caif_device_lora_adapter.h"
#include "caif/caif_device_gated_activations.h"
#include "caif/caif_safetensors_format.h"
#include "caif/caif_cuda_kernels.h"
#include "caif/caif_constants.h"

#include "ise_lib/ise_out.h"

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>

using namespace instance;

//------------------------------------------------------------------------------
// ParseConfig
//------------------------------------------------------------------------------
RTNR_GLMConfig_t RTNR_GLMModelBuilder::ParseConfig(const std::string &config_json_path)
{
  try
  {
    // Read the file
    std::ifstream file(config_json_path);
    if(file.is_open()==false)
    {
      THROW_RTNRE("Failed to open config.json");
    }

    std::ostringstream ss;
    ss<<file.rdbuf();
    std::string json_str=ss.str();
    file.close();

    // Parse with rapidjson
    rapidjson::Document doc;
    doc.Parse(json_str.c_str());

    if(doc.HasParseError()==true)
    {
      THROW_RTNRE("Failed to parse config.json");
    }

    RTNR_GLMConfig_t config;

    // Required fields
    if(doc.HasMember("vocab_size")==true)
    {
      config.vocab_size=doc["vocab_size"].GetUint();
    }
    else
    {
      THROW_RTNRE("config.json missing vocab_size");
    }

    if(doc.HasMember("hidden_size")==true)
    {
      config.dim=doc["hidden_size"].GetUint();
    }
    else
    {
      THROW_RTNRE("config.json missing hidden_size");
    }

    if(doc.HasMember("num_hidden_layers")==true)
    {
      config.num_layers=doc["num_hidden_layers"].GetUint();
    }
    else
    {
      THROW_RTNRE("config.json missing num_hidden_layers");
    }

    if(doc.HasMember("num_attention_heads")==true)
    {
      config.num_heads=doc["num_attention_heads"].GetUint();
    }
    else
    {
      THROW_RTNRE("config.json missing num_attention_heads");
    }

    // MLA dimensions
    if(doc.HasMember("q_lora_rank")==true)
    {
      config.q_lora_rank=doc["q_lora_rank"].GetUint();
    }
    else
    {
      config.q_lora_rank=0;
    }

    if(doc.HasMember("kv_lora_rank")==true)
    {
      config.kv_lora_rank=doc["kv_lora_rank"].GetUint();
    }
    else
    {
      config.kv_lora_rank=0;
    }

    if(doc.HasMember("qk_rope_head_dim")==true)
    {
      config.qk_rope_head_dim=doc["qk_rope_head_dim"].GetUint();
    }
    else
    {
      config.qk_rope_head_dim=64;
    }

    if(doc.HasMember("qk_nope_head_dim")==true)
    {
      config.qk_nope_head_dim=doc["qk_nope_head_dim"].GetUint();
    }
    else
    {
      config.qk_nope_head_dim=192;
    }

    if(doc.HasMember("v_head_dim")==true)
    {
      config.v_head_dim=doc["v_head_dim"].GetUint();
    }
    else
    {
      config.v_head_dim=256;
    }

    // RoPE and norm
    if(doc.HasMember("rope_theta")==true)
    {
      config.rope_base=static_cast<float>(doc["rope_theta"].GetDouble());
    }
    else
    {
      config.rope_base=1000000.0f;
    }

    if(doc.HasMember("rms_norm_eps")==true)
    {
      config.rms_norm_eps=static_cast<float>(doc["rms_norm_eps"].GetDouble());
    }
    else
    {
      config.rms_norm_eps=1e-5f;
    }

    // Dense layer index
    if(doc.HasMember("first_k_dense_replace")==true)
    {
      config.dense_layer_index=doc["first_k_dense_replace"].GetUint();
    }
    else
    {
      config.dense_layer_index=0;
    }

    // FFN dim (for dense layer)
    if(doc.HasMember("intermediate_size")==true)
    {
      config.ffn_dim=doc["intermediate_size"].GetUint();
    }
    else
    {
      config.ffn_dim=config.dim*4;
    }

    // MoE config
    if(doc.HasMember("num_experts")==true)
    {
      config.moe_num_experts=doc["num_experts"].GetUint();
    }
    else if(doc.HasMember("n_routed_experts")==true)
    {
      config.moe_num_experts=doc["n_routed_experts"].GetUint();
    }
    else
    {
      config.moe_num_experts=64;
    }

    if(doc.HasMember("num_experts_per_tok")==true)
    {
      config.moe_top_k=doc["num_experts_per_tok"].GetUint();
    }
    else if(doc.HasMember("top_k")==true)
    {
      config.moe_top_k=doc["top_k"].GetUint();
    }
    else
    {
      config.moe_top_k=8;
    }

    if(doc.HasMember("moe_intermediate_size")==true)
    {
      config.moe_hidden_dim=doc["moe_intermediate_size"].GetUint();
    }
    else
    {
      config.moe_hidden_dim=config.ffn_dim;
    }

    if(doc.HasMember("n_shared_experts")==true)
    {
      config.moe_shared_experts=doc["n_shared_experts"].GetUint();
    }
    else
    {
      config.moe_shared_experts=0;
    }

    // Weight tying
    if(doc.HasMember("tie_word_embeddings")==true)
    {
      config.tie_word_embeddings=doc["tie_word_embeddings"].GetBool();
    }
    else
    {
      config.tie_word_embeddings=false;
    }

    ISE_Out::Out()<<"GLM config: vocab="<<config.vocab_size
                  <<" dim="<<config.dim
                  <<" layers="<<config.num_layers
                  <<" heads="<<config.num_heads
                  <<" experts="<<config.moe_num_experts
                  <<" top_k="<<config.moe_top_k
                  <<" shared="<<config.moe_shared_experts
                  <<std::endl;

    return config;
  }
  RTNR_CATCH_BLOCK("RTNR_GLMModelBuilder::ParseConfig")
}

//------------------------------------------------------------------------------
// IsLoRATarget
//------------------------------------------------------------------------------
bool RTNR_GLMModelBuilder::IsLoRATarget(const std::string &name,
                                        const std::vector<std::string> &targets)
{
  for(size_t i=0;i<targets.size();++i)
  {
    if(targets[i]==name)
    {
      return true;
    }
  }
  return false;
}

//------------------------------------------------------------------------------
// MakeProjection
//------------------------------------------------------------------------------
std::unique_ptr<CAIF_DeviceLayer> RTNR_GLMModelBuilder::MakeProjection(
                                     uint32_t input_dim,
                                     uint32_t output_dim,
                                     CAIF_DataType::CAIF_DataType_e dtype,
                                     CAIF_CudaStream &stream,
                                     const std::string &proj_name,
                                     const std::string &hf_weight_name,
                                     uint32_t lora_rank,
                                     float lora_alpha,
                                     const std::vector<std::string> &lora_targets)
{
  try
  {
    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear>(input_dim,
                                                         output_dim,
                                                         dtype,
                                                         stream,
                                                         g_caif_quant_default_group_size,
                                                         false);

    CAIF_DeviceFrozenLinear *raw=frozen.get();

    if(lora_rank>0&&IsLoRATarget(proj_name,lora_targets)==true)
    {
      CAIF_DeviceLoRAAdapter::LoRAConfig_t lora_cfg;
      lora_cfg.rank=lora_rank;
      lora_cfg.alpha=lora_alpha;
      lora_cfg.input_dim=input_dim;
      lora_cfg.output_dim=output_dim;

      _weight_map[hf_weight_name]=raw;
      return std::make_unique<CAIF_DeviceLoRAAdapter>(lora_cfg,std::move(frozen),stream);
    }

    _weight_map[hf_weight_name]=raw;
    return frozen;
  }
  RTNR_CATCH_BLOCK("RTNR_GLMModelBuilder::MakeProjection")
}

//------------------------------------------------------------------------------
// BuildModel
//------------------------------------------------------------------------------
void RTNR_GLMModelBuilder::BuildModel(CAIF_DeviceNetwork &network,
                                      CAIF_CudaStream &stream,
                                      const RTNR_GLMConfig_t &config,
                                      CAIF_DataType::CAIF_DataType_e storage_dtype,
                                      uint32_t lora_rank,
                                      float lora_alpha,
                                      const std::vector<std::string> &lora_targets)
{
  try
  {
    ISE_Out::Out()<<"Building GLM model: "<<config.num_layers<<" layers, dtype="
                  <<CAIF_DataType(storage_dtype).Name()<<std::endl;

    // Layer 0: Token embedding
    CAIF_DeviceTokenEmbedding::Config_t emb_config;
    emb_config.vocab_size=config.vocab_size;
    emb_config.dim=config.dim;
    network.AddLayer(std::make_unique<CAIF_DeviceTokenEmbedding>(emb_config,stream));

    // MLA config (shared by all transformer layers)
    CAIF_DeviceMLAttention::MLAConfig_t mla_config;
    mla_config.dim=config.dim;
    mla_config.num_heads=config.num_heads;
    mla_config.q_lora_rank=config.q_lora_rank;
    mla_config.kv_lora_rank=config.kv_lora_rank;
    mla_config.qk_rope_head_dim=config.qk_rope_head_dim;
    mla_config.qk_nope_head_dim=config.qk_nope_head_dim;
    mla_config.v_head_dim=config.v_head_dim;
    mla_config.causal=true;
    mla_config.rope_base=config.rope_base;
    mla_config.rms_norm_eps=config.rms_norm_eps;

    // Derived MLA dimensions
    uint32_t qk_head_dim=config.qk_nope_head_dim+config.qk_rope_head_dim;
    uint32_t q_proj_dim=config.num_heads*qk_head_dim;
    uint32_t kv_compress_dim=config.kv_lora_rank+config.qk_rope_head_dim;
    uint32_t kv_decomp_dim=config.num_heads*(config.qk_nope_head_dim+config.v_head_dim);
    uint32_t o_input_dim=config.num_heads*config.v_head_dim;

    // Layers 1 through num_layers: PreNormBlocks
    _weight_map.clear();

    for(uint32_t layer_idx=0;layer_idx<config.num_layers;++layer_idx)
    {
      std::string hf="model.layers."+std::to_string(layer_idx)+".";

      // SubLayer 0: (RMSNorm, MLA with projections)
      CAIF_DeviceMLAttention::MLAProjections_t mla_proj;
      mla_proj.q_compress=MakeProjection(config.dim,
                                         config.q_lora_rank,
                                         storage_dtype,
                                         stream,
                                         "q",
                                         hf+"self_attn.q_a_proj.weight",
                                         lora_rank,
                                         lora_alpha,
                                         lora_targets);
      mla_proj.q_decompress=MakeProjection(config.q_lora_rank,
                                            q_proj_dim,
                                            storage_dtype,
                                            stream,
                                            "q",
                                            hf+"self_attn.q_b_proj.weight",
                                            lora_rank,
                                            lora_alpha,
                                            lora_targets);
      mla_proj.kv_compress=MakeProjection(config.dim,
                                           kv_compress_dim,
                                           storage_dtype,
                                           stream,
                                           "kv",
                                           hf+"self_attn.kv_a_proj_with_mqa.weight",
                                           lora_rank,
                                           lora_alpha,
                                           lora_targets);
      mla_proj.kv_decompress=MakeProjection(config.kv_lora_rank,
                                             kv_decomp_dim,
                                             storage_dtype,
                                             stream,
                                             "kv",
                                             hf+"self_attn.kv_b_proj.weight",
                                             lora_rank,
                                             lora_alpha,
                                             lora_targets);
      mla_proj.o_proj=MakeProjection(o_input_dim,
                                      config.dim,
                                      storage_dtype,
                                      stream,
                                      "o",
                                      hf+"self_attn.o_proj.weight",
                                      lora_rank,
                                      lora_alpha,
                                      lora_targets);

      auto mla=std::make_unique<CAIF_DeviceMLAttention>(mla_config,
                                                        std::move(mla_proj),
                                                        stream);

      auto attn_norm=std::make_unique<CAIF_DeviceRMSNorm>(config.dim,
                                                          stream,
                                                          config.rms_norm_eps);

      // SubLayer 1: (RMSNorm, FFN or MoE)
      // first_k_dense_replace: first K layers use dense FFN, rest use MoE
      std::unique_ptr<CAIF_DeviceLayer> ffn_or_moe;
      bool is_dense=(layer_idx<config.dense_layer_index);

      if(layer_idx==0&&config.dense_layer_index==0&&config.ffn_dim>0)
      {
        is_dense=true;
      }

      if(is_dense==true)
      {
        // Dense FFN with SwiGLU
        CAIF_DeviceFFN::FFNConfig_t ffn_cfg;
        ffn_cfg.dim=config.dim;
        ffn_cfg.ffn_dim=config.ffn_dim;

        CAIF_DeviceFFN::FFNProjections_t ffn_proj;
        ffn_proj.gate=MakeProjection(config.dim,
                                     config.ffn_dim,
                                     storage_dtype,
                                     stream,
                                     "gate",
                                     hf+"mlp.gate_proj.weight",
                                     lora_rank,
                                     lora_alpha,
                                     lora_targets);
        ffn_proj.up=MakeProjection(config.dim,
                                   config.ffn_dim,
                                   storage_dtype,
                                   stream,
                                   "up",
                                   hf+"mlp.up_proj.weight",
                                   lora_rank,
                                   lora_alpha,
                                   lora_targets);
        ffn_proj.down=MakeProjection(config.ffn_dim,
                                     config.dim,
                                     storage_dtype,
                                     stream,
                                     "down",
                                     hf+"mlp.down_proj.weight",
                                     lora_rank,
                                     lora_alpha,
                                     lora_targets);

        auto activation=std::make_unique<CAIF_DeviceSwiGLUActivation>();
        ffn_or_moe=std::make_unique<CAIF_DeviceFFN>(ffn_cfg,
                                                   std::move(ffn_proj),
                                                   std::move(activation),
                                                   stream);
      }
      else
      {
        // MoE layer with pre-built experts
        CAIF_DeviceMoEExpert::Config_t expert_cfg;
        expert_cfg.input_dim=config.dim;
        expert_cfg.hidden_dim=config.moe_hidden_dim;
        expert_cfg.use_gated=true;
        expert_cfg.use_bias=false;

        std::vector<std::unique_ptr<CAIF_DeviceMoEExpert>> routed_experts;
        routed_experts.reserve(config.moe_num_experts);
        for(uint32_t e=0;e<config.moe_num_experts;++e)
        {
          std::string ep_hf=hf+"mlp.experts."+std::to_string(e)+".";

          CAIF_DeviceMoEExpert::MoEExpertProjections_t ep;
          ep.gate=MakeProjection(config.dim,
                                 config.moe_hidden_dim,
                                 storage_dtype,
                                 stream,
                                 "gate",
                                 ep_hf+"gate_proj.weight",
                                 lora_rank,
                                 lora_alpha,
                                 lora_targets);
          ep.up=MakeProjection(config.dim,
                               config.moe_hidden_dim,
                               storage_dtype,
                               stream,
                               "up",
                               ep_hf+"up_proj.weight",
                               lora_rank,
                               lora_alpha,
                               lora_targets);
          ep.down=MakeProjection(config.moe_hidden_dim,
                                 config.dim,
                                 storage_dtype,
                                 stream,
                                 "down",
                                 ep_hf+"down_proj.weight",
                                 lora_rank,
                                 lora_alpha,
                                 lora_targets);

          routed_experts.push_back(
              std::make_unique<CAIF_DeviceMoEExpert>(expert_cfg,
                                                    std::move(ep),
                                                    stream));
        }

        // Shared expert (single FFN with scaled hidden dim)
        std::vector<std::unique_ptr<CAIF_DeviceMoEExpert>> shared_experts;
        if(config.moe_shared_experts>0)
        {
          CAIF_DeviceMoEExpert::Config_t shared_cfg;
          shared_cfg.input_dim=config.dim;
          shared_cfg.hidden_dim=config.moe_shared_experts*config.moe_hidden_dim;
          shared_cfg.use_gated=true;
          shared_cfg.use_bias=false;

          std::string se_hf=hf+"mlp.shared_experts.";

          CAIF_DeviceMoEExpert::MoEExpertProjections_t sp;
          sp.gate=MakeProjection(config.dim,
                                 shared_cfg.hidden_dim,
                                 storage_dtype,
                                 stream,
                                 "gate",
                                 se_hf+"gate_proj.weight",
                                 lora_rank,
                                 lora_alpha,
                                 lora_targets);
          sp.up=MakeProjection(config.dim,
                               shared_cfg.hidden_dim,
                               storage_dtype,
                               stream,
                               "up",
                               se_hf+"up_proj.weight",
                               lora_rank,
                               lora_alpha,
                               lora_targets);
          sp.down=MakeProjection(shared_cfg.hidden_dim,
                                 config.dim,
                                 storage_dtype,
                                 stream,
                                 "down",
                                 se_hf+"down_proj.weight",
                                 lora_rank,
                                 lora_alpha,
                                 lora_targets);

          shared_experts.push_back(
              std::make_unique<CAIF_DeviceMoEExpert>(shared_cfg,
                                                    std::move(sp),
                                                    stream));
        }

        CAIF_DeviceMoELayer::Config_t moe_cfg;
        moe_cfg.input_dim=config.dim;
        moe_cfg.hidden_dim=config.moe_hidden_dim;
        moe_cfg.num_experts=config.moe_num_experts;
        moe_cfg.top_k=config.moe_top_k;
        moe_cfg.expert_use_gated=true;
        moe_cfg.expert_use_bias=false;
        moe_cfg.num_shared_experts=config.moe_shared_experts;
        moe_cfg.shared_hidden_dim=0;
        moe_cfg.fine_grained=false;
        moe_cfg.fine_grained_factor=1;
        moe_cfg.router_use_bias=false;
        moe_cfg.router_noise_std=0.0f;
        moe_cfg.capacity_factor=1.5f;
        moe_cfg.overflow_strategy=CAIF_DeviceMoELayer::OverflowStrategy_e::Drop;
        moe_cfg.balance_loss_weight=0.01f;
        moe_cfg.z_loss_weight=0.001f;

        ffn_or_moe=std::make_unique<CAIF_DeviceMoELayer>(moe_cfg,
                                                         std::move(routed_experts),
                                                         std::move(shared_experts),
                                                         stream);
      }

      auto ffn_norm=std::make_unique<CAIF_DeviceRMSNorm>(config.dim,
                                                         stream,
                                                         config.rms_norm_eps);

      // Assemble PreNormBlock with 2 sublayers
      CAIF_DevicePreNormBlock::SubLayerVec_t sub_layers;
      sub_layers.reserve(2);

      CAIF_DevicePreNormBlock::SubLayer_t attn_sub;
      attn_sub.norm_prefix="input_layernorm.";
      attn_sub.layer_prefix="self_attn.";
      attn_sub.norm=std::move(attn_norm);
      attn_sub.layer=std::move(mla);
      sub_layers.push_back(std::move(attn_sub));

      CAIF_DevicePreNormBlock::SubLayer_t ffn_sub;
      ffn_sub.norm_prefix="post_attention_layernorm.";
      ffn_sub.layer_prefix="mlp.";
      ffn_sub.norm=std::move(ffn_norm);
      ffn_sub.layer=std::move(ffn_or_moe);
      sub_layers.push_back(std::move(ffn_sub));

      network.AddLayer(std::make_unique<CAIF_DevicePreNormBlock>(std::move(sub_layers),stream));

      if((layer_idx+1)%10==0||layer_idx==config.num_layers-1)
      {
        ISE_Out::Out()<<"  Built layer "
                      <<(layer_idx+1)
                      <<"/"
                      <<config.num_layers
                      <<std::endl;
      }
    }

    // Layer N+1: Final RMSNorm
    network.AddLayer(std::make_unique<CAIF_DeviceRMSNorm>(config.dim,
                                                         stream,
                                                         config.rms_norm_eps));

    // Layer N+2: Linear head (lm_head)
    CAIF_DeviceLinearHead::Config_t head_cfg;
    head_cfg.input_dim=config.dim;
    head_cfg.output_dim=config.vocab_size;
    head_cfg.use_bias=false;

    if(config.tie_word_embeddings==true)
    {
      // Weight-tied: share embedding table
      auto &emb_layer=network.Layer(0);
      network.AddLayer(std::make_unique<CAIF_DeviceLinearHead>(head_cfg,
                                                              emb_layer.ParameterTensor(0),
                                                              emb_layer.GradientTensor(0),
                                                              stream));
    }
    else
    {
      network.AddLayer(std::make_unique<CAIF_DeviceLinearHead>(head_cfg,stream));
    }

    ISE_Out::Out()<<"GLM model built: "
                  <<network.LayerCount()
                  <<" layers, "
                  <<network.TotalParameterCount()
                  <<" total parameters"
                  <<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_GLMModelBuilder::BuildModel")
}

//------------------------------------------------------------------------------
// LoadWeights
//------------------------------------------------------------------------------
void RTNR_GLMModelBuilder::LoadWeights(CAIF_DeviceNetwork &network,
                                       CAIF_CudaStream &stream,
                                       const std::string &model_dir,
                                       const RTNR_GLMConfig_t &config,
                                       CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    CAIF_SafeTensorsFormat st;

    ISE_Out::Out()<<"Loading weights from "<<model_dir<<std::endl;

    // Layer 0: Token embedding
    CAIF_DeviceTensor emb_w=st.LoadTensorByName(model_dir,"model.embed_tokens.weight",stream);
    network.Layer(0).ParameterTensor(0)=emb_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
    ISE_Out::Out()<<"  Loaded embedding"<<std::endl;

    // Load all FrozenLinear projection weights via the weight map
    uint32_t proj_count=0;
    for(auto &[hf_name,frozen]:_weight_map)
    {
      CAIF_DeviceTensor raw_w=st.LoadTensorByName(model_dir,hf_name,stream);

      if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        frozen->LoadFromTensor(raw_w.To(CAIF_DataType::CAIF_DataType_e::Float32));
      }
      else if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float16||
              storage_dtype==CAIF_DataType::CAIF_DataType_e::BFloat16)
      {
        frozen->LoadFromTensor(raw_w.To(storage_dtype));
      }
      else if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Int8)
      {
        frozen->LoadFromTensor(raw_w.To(CAIF_DataType::CAIF_DataType_e::Int8));
      }
      else if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Int4)
      {
        CAIF_DeviceTensor fp32_w=raw_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
        uint32_t rows=fp32_w.Shape()[0];
        uint32_t cols=fp32_w.Shape()[1];
        uint32_t num_groups=(rows*cols+g_caif_quant_default_group_size-1)/
                            g_caif_quant_default_group_size;

        CAIF_DeviceTensor int4_w=CAIF_DeviceTensor::Uninitialized({(rows*cols+1)/2},
                                                                stream,
                                                                CAIF_DataType::CAIF_DataType_e::UInt8);
        CAIF_DeviceTensor scales=CAIF_DeviceTensor::Uninitialized({num_groups},
                                                                 stream,
                                                                 CAIF_DataType::CAIF_DataType_e::Float16);

        launch_quantize_to_int4(fp32_w.DevicePtr(),
                                 int4_w.DeviceDataRaw(),
                                 scales.DeviceDataRaw(),
                                 rows*cols,
                                 g_caif_quant_default_group_size,
                                 stream.Handle());

        frozen->LoadFromTensor(std::move(int4_w));

        // Copy scales to host for LoadScalesFromHost
        std::vector<uint8_t> host_scales(num_groups*2);
        scales.CopyToHostRaw(host_scales.data());
        frozen->LoadScalesFromHost(host_scales.data(),num_groups*2);
      }
      else
      {
        frozen->LoadFromTensor(raw_w.To(storage_dtype));
      }

      ++proj_count;
    }
    ISE_Out::Out()<<"  Loaded "<<proj_count<<" projection weights"<<std::endl;

    // Transformer layer norms (small FP32 tensors loaded by parameter index)
    for(uint32_t layer_idx=0;layer_idx<config.num_layers;++layer_idx)
    {
      std::string prefix="model.layers."+std::to_string(layer_idx)+".";
      auto &block=network.Layer(layer_idx+1);

      // Input layernorm (sublayer 0 norm gamma = parameter index 0)
      std::string norm_name=prefix+"input_layernorm.weight";
      CAIF_DeviceTensor norm_w=st.LoadTensorByName(model_dir,norm_name,stream);
      block.ParameterTensor(0)=norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);

      // MLA internal norms: q_a_layernorm and kv_a_layernorm
      // These are exposed as named parameters within the PreNormBlock
      norm_name=prefix+"self_attn.q_a_layernorm.weight";
      norm_w=st.LoadTensorByName(model_dir,norm_name,stream);

      std::vector<std::string> names=block.ParameterNames("");
      for(size_t i=0;i<names.size();++i)
      {
        if(names[i].find("q_a_layernorm.")!=std::string::npos)
        {
          block.ParameterTensor(i)=norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
          break;
        }
      }

      norm_name=prefix+"self_attn.kv_a_layernorm.weight";
      norm_w=st.LoadTensorByName(model_dir,norm_name,stream);

      for(size_t i=0;i<names.size();++i)
      {
        if(names[i].find("kv_a_layernorm.")!=std::string::npos)
        {
          block.ParameterTensor(i)=norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
          break;
        }
      }

      // Post-attention layernorm
      norm_name=prefix+"post_attention_layernorm.weight";
      norm_w=st.LoadTensorByName(model_dir,norm_name,stream);

      for(size_t i=0;i<names.size();++i)
      {
        if(names[i].find("post_attention_layernorm.")!=std::string::npos)
        {
          block.ParameterTensor(i)=norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
          break;
        }
      }

      // MoE router weight (for non-dense layers)
      bool is_dense=(layer_idx<config.dense_layer_index);
      if(layer_idx==0&&config.dense_layer_index==0&&config.ffn_dim>0)
      {
        is_dense=true;
      }

      if(is_dense==false)
      {
        norm_name=prefix+"mlp.gate.weight";
        norm_w=st.LoadTensorByName(model_dir,norm_name,stream);

        for(size_t i=0;i<names.size();++i)
        {
          if(names[i].find("mlp.")!=std::string::npos&&
             names[i].find("gate.weight")!=std::string::npos&&
             names[i].find("gate_proj")==std::string::npos)
          {
            block.ParameterTensor(i)=norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
            break;
          }
        }
      }

      if((layer_idx+1)%10==0||layer_idx==config.num_layers-1)
      {
        ISE_Out::Out()<<"  Loaded norms for layer "
                      <<(layer_idx+1)
                      <<"/"
                      <<config.num_layers
                      <<std::endl;
      }
    }

    // Final RMSNorm
    CAIF_DeviceTensor final_norm_w=st.LoadTensorByName(model_dir,"model.norm.weight",stream);
    size_t final_norm_idx=config.num_layers+1;
    network.Layer(final_norm_idx).ParameterTensor(0)=final_norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
    ISE_Out::Out()<<"  Loaded final norm"<<std::endl;

    // lm_head (if not tied)
    // HF lm_head.weight is [vocab_size, dim], LinearHead expects [dim, vocab_size]
    if(config.tie_word_embeddings==false)
    {
      CAIF_DeviceTensor head_w=st.LoadTensorByName(model_dir,"lm_head.weight",stream);
      CAIF_DeviceTensor head_w_fp32=head_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
      CAIF_DeviceTensor head_w_t=CAIF_DeviceTensor::Uninitialized({config.dim,config.vocab_size},
                                                                 stream);
      CAIF_DeviceOps::Transpose(head_w_fp32,head_w_t);
      size_t head_idx=config.num_layers+2;
      network.Layer(head_idx).ParameterTensor(0)=std::move(head_w_t);
      ISE_Out::Out()<<"  Loaded lm_head"<<std::endl;
    }

    ISE_Out::Out()<<"Weight loading complete"<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_GLMModelBuilder::LoadWeights")
}

//------------------------------------------------------------------------------
// SaveLoRAWeights
//------------------------------------------------------------------------------
void RTNR_GLMModelBuilder::SaveLoRAWeights(const CAIF_DeviceNetwork &network,
                                           const std::string &path)
{
  try
  {
    std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> tensors;

    // Iterate all layers, collect LoRA parameters
    for(size_t layer_idx=0;layer_idx<network.LayerCount();++layer_idx)
    {
      const auto &layer=network.Layer(layer_idx);
      std::string layer_prefix="layers."+std::to_string(layer_idx)+".";

      std::vector<std::string> names=layer.ParameterNames(layer_prefix);

      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        // LoRA parameters have "lora_a" or "lora_b" in their names
        if(names[p].find("lora_a")!=std::string::npos||
           names[p].find("lora_b")!=std::string::npos)
        {
          tensors.push_back({names[p],&layer.ParameterTensor(p)});
        }
      }
    }

    if(tensors.empty()==true)
    {
      ISE_Out::Out()<<"No LoRA parameters found to save"<<std::endl;
      return;
    }

    std::map<std::string,std::string> metadata;
    metadata["format"]="lora";
    metadata["framework"]="caif";

    CAIF_SafeTensorsFormat st;
    st.Save(path,tensors,metadata);

    ISE_Out::Out()<<"Saved "<<tensors.size()<<" LoRA tensors to "<<path<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_GLMModelBuilder::SaveLoRAWeights")
}

//------------------------------------------------------------------------------
// LoadLoRAWeights
//------------------------------------------------------------------------------
void RTNR_GLMModelBuilder::LoadLoRAWeights(CAIF_DeviceNetwork &network,
                                           const std::string &path,
                                           CAIF_CudaStream &stream)
{
  try
  {
    CAIF_SafeTensorsFormat st;
    std::map<std::string,CAIF_DeviceTensor> loaded=st.Load(path,stream);

    uint32_t loaded_count=0;

    // Match loaded tensors to network parameters by name
    for(size_t layer_idx=0;layer_idx<network.LayerCount();++layer_idx)
    {
      auto &layer=network.Layer(layer_idx);
      std::string layer_prefix="layers."+std::to_string(layer_idx)+".";

      std::vector<std::string> names=layer.ParameterNames(layer_prefix);

      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        auto it=loaded.find(names[p]);
        if(it!=loaded.end())
        {
          CAIF_DeviceTensor fp32=it->second.To(CAIF_DataType::CAIF_DataType_e::Float32);
          layer.ParameterTensor(p)=std::move(fp32);
          ++loaded_count;
        }
      }
    }

    ISE_Out::Out()<<"Loaded "<<loaded_count<<" LoRA tensors from "<<path<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_GLMModelBuilder::LoadLoRAWeights")
}
