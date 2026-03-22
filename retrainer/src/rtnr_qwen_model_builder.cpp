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
// Retrainer - Qwen2.5-Coder-1.5B Model Builder
//------------------------------------------------------------------------------
#include "rtnr_qwen_model_builder.h"
#include "rtnr_constants.h"

#include "caif/caif_device_token_embedding.h"
#include "caif/caif_device_pre_norm_block.h"
#include "caif/caif_device_multi_head_attention.h"
#include "caif/caif_device_ffn.h"
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
RTNR_QwenConfig_t RTNR_QwenModelBuilder::ParseConfig(
                                           const std::string &config_json_path)
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

    RTNR_QwenConfig_t config;

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

    // KV heads (GQA)
    if(doc.HasMember("num_key_value_heads")==true)
    {
      config.num_kv_heads=doc["num_key_value_heads"].GetUint();
    }
    else
    {
      config.num_kv_heads=config.num_heads;
    }

    // Head dim: explicit field or computed
    if(doc.HasMember("head_dim")==true)
    {
      config.head_dim=doc["head_dim"].GetUint();
    }
    else
    {
      config.head_dim=config.dim/config.num_heads;
    }

    // FFN intermediate size
    if(doc.HasMember("intermediate_size")==true)
    {
      config.ffn_dim=doc["intermediate_size"].GetUint();
    }
    else
    {
      config.ffn_dim=config.dim*4;
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
      config.rms_norm_eps=
          static_cast<float>(doc["rms_norm_eps"].GetDouble());
    }
    else
    {
      config.rms_norm_eps=1e-6f;
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

    // Attention bias
    if(doc.HasMember("attention_bias")==true)
    {
      config.use_qkv_bias=doc["attention_bias"].GetBool();
    }
    else
    {
      config.use_qkv_bias=true;
    }

    ISE_Out::Out()<<"Qwen config: vocab="
                  <<config.vocab_size
                  <<" dim="
                  <<config.dim
                  <<" layers="
                  <<config.num_layers
                  <<" heads="
                  <<config.num_heads
                  <<" kv_heads="
                  <<config.num_kv_heads
                  <<" head_dim="
                  <<config.head_dim
                  <<" ffn="
                  <<config.ffn_dim
                  <<" bias="
                  <<config.use_qkv_bias
                  <<std::endl;

    return config;
  }
  RTNR_CATCH_BLOCK("RTNR_QwenModelBuilder::ParseConfig")
}

//------------------------------------------------------------------------------
// IsLoRATarget
//------------------------------------------------------------------------------
bool RTNR_QwenModelBuilder::IsLoRATarget(
                                 const std::string &name,
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
std::unique_ptr<CAIF_DeviceLayer> RTNR_QwenModelBuilder::MakeProjection(
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
    auto frozen=std::make_unique<CAIF_DeviceFrozenLinear>(
                    input_dim,
                    output_dim,
                    dtype,
                    stream,
                    g_caif_quant_default_group_size,
                    false);

    CAIF_DeviceFrozenLinear *raw=frozen.get();

    if(lora_rank>0 && IsLoRATarget(proj_name,lora_targets)==true)
    {
      CAIF_DeviceLoRAAdapter::LoRAConfig_t lora_cfg;
      lora_cfg.rank=lora_rank;
      lora_cfg.alpha=lora_alpha;
      lora_cfg.input_dim=input_dim;
      lora_cfg.output_dim=output_dim;

      _weight_map[hf_weight_name]=raw;
      return std::make_unique<CAIF_DeviceLoRAAdapter>(lora_cfg,
                                                      std::move(frozen),
                                                      stream);
    }

    _weight_map[hf_weight_name]=raw;
    return frozen;
  }
  RTNR_CATCH_BLOCK("RTNR_QwenModelBuilder::MakeProjection")
}

//------------------------------------------------------------------------------
// BuildModel
//------------------------------------------------------------------------------
void RTNR_QwenModelBuilder::BuildModel(
                                CAIF_DeviceNetwork &network,
                                CAIF_CudaStream &stream,
                                const RTNR_QwenConfig_t &config,
                                CAIF_DataType::CAIF_DataType_e storage_dtype,
                                uint32_t lora_rank,
                                float lora_alpha,
                                const std::vector<std::string> &lora_targets)
{
  try
  {
    ISE_Out::Out()<<"Building Qwen model: "
                  <<config.num_layers
                  <<" layers, dtype="
                  <<CAIF_DataType(storage_dtype).Name()
                  <<std::endl;

    // Layer 0: Token embedding
    CAIF_DeviceTokenEmbedding::Config_t emb_config;
    emb_config.vocab_size=config.vocab_size;
    emb_config.dim=config.dim;
    network.AddLayer(
        std::make_unique<CAIF_DeviceTokenEmbedding>(emb_config,stream));

    // MHA config (shared by all transformer layers)
    CAIF_DeviceMultiHeadAttention::AttentionConfig_t attn_config;
    attn_config.dim=config.dim;
    attn_config.num_heads=config.num_heads;
    attn_config.num_kv_heads=config.num_kv_heads;
    attn_config.head_dim=config.head_dim;
    attn_config.causal=true;
    attn_config.use_rope=true;
    attn_config.rope_base=config.rope_base;
    attn_config.dropout_rate=0.0f;

    // Derived dimensions
    uint32_t qk_dim=config.num_heads*config.head_dim;
    uint32_t kv_dim=config.num_kv_heads*config.head_dim;

    // Layers 1 through num_layers: PreNormBlocks
    _weight_map.clear();
    _mha_layers.clear();
    _mha_layers.reserve(config.num_layers);

    for(uint32_t layer_idx=0;layer_idx<config.num_layers;++layer_idx)
    {
      std::string hf="model.layers."+std::to_string(layer_idx)+".";

      // SubLayer 0: (RMSNorm, MHA with Q/K/V/O projections)
      CAIF_DeviceMultiHeadAttention::MHAProjections_t mha_proj;
      mha_proj.q_proj=MakeProjection(config.dim,
                                      qk_dim,
                                      storage_dtype,
                                      stream,
                                      "q",
                                      hf+"self_attn.q_proj.weight",
                                      lora_rank,
                                      lora_alpha,
                                      lora_targets);
      mha_proj.k_proj=MakeProjection(config.dim,
                                      kv_dim,
                                      storage_dtype,
                                      stream,
                                      "k",
                                      hf+"self_attn.k_proj.weight",
                                      lora_rank,
                                      lora_alpha,
                                      lora_targets);
      mha_proj.v_proj=MakeProjection(config.dim,
                                      kv_dim,
                                      storage_dtype,
                                      stream,
                                      "v",
                                      hf+"self_attn.v_proj.weight",
                                      lora_rank,
                                      lora_alpha,
                                      lora_targets);
      mha_proj.o_proj=MakeProjection(config.dim,
                                      config.dim,
                                      storage_dtype,
                                      stream,
                                      "o",
                                      hf+"self_attn.o_proj.weight",
                                      lora_rank,
                                      lora_alpha,
                                      lora_targets);

      // Allocate bias tensors as FP32 zeros if model uses QKV bias
      if(config.use_qkv_bias==true)
      {
        mha_proj.q_bias=CAIF_DeviceTensor::Zeros({qk_dim},stream);
        mha_proj.k_bias=CAIF_DeviceTensor::Zeros({kv_dim},stream);
        mha_proj.v_bias=CAIF_DeviceTensor::Zeros({kv_dim},stream);
      }

      auto mha=std::make_unique<CAIF_DeviceMultiHeadAttention>(
                    attn_config,
                    std::move(mha_proj),
                    stream);

      // Store raw MHA pointer for bias loading in LoadWeights
      CAIF_DeviceMultiHeadAttention *mha_raw=mha.get();
      _mha_layers.push_back(mha_raw);

      auto attn_norm=std::make_unique<CAIF_DeviceRMSNorm>(
                         config.dim,
                         stream,
                         config.rms_norm_eps);

      // SubLayer 1: (RMSNorm, FFN with SwiGLU)
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
      auto ffn=std::make_unique<CAIF_DeviceFFN>(ffn_cfg,
                                                std::move(ffn_proj),
                                                std::move(activation),
                                                stream);

      auto ffn_norm=std::make_unique<CAIF_DeviceRMSNorm>(
                        config.dim,
                        stream,
                        config.rms_norm_eps);

      // Assemble PreNormBlock with 2 sublayers
      CAIF_DevicePreNormBlock::SubLayerVec_t sub_layers;
      sub_layers.reserve(2);

      CAIF_DevicePreNormBlock::SubLayer_t attn_sub;
      attn_sub.norm_prefix="input_layernorm.";
      attn_sub.layer_prefix="self_attn.";
      attn_sub.norm=std::move(attn_norm);
      attn_sub.layer=std::move(mha);
      sub_layers.push_back(std::move(attn_sub));

      CAIF_DevicePreNormBlock::SubLayer_t ffn_sub;
      ffn_sub.norm_prefix="post_attention_layernorm.";
      ffn_sub.layer_prefix="mlp.";
      ffn_sub.norm=std::move(ffn_norm);
      ffn_sub.layer=std::move(ffn);
      sub_layers.push_back(std::move(ffn_sub));

      network.AddLayer(
          std::make_unique<CAIF_DevicePreNormBlock>(
              std::move(sub_layers),stream));

      if((layer_idx+1)%10==0 || layer_idx==config.num_layers-1)
      {
        ISE_Out::Out()<<"  Built layer "
                      <<(layer_idx+1)
                      <<"/"
                      <<config.num_layers
                      <<std::endl;
      }
    }

    // Layer N+1: Final RMSNorm
    network.AddLayer(
        std::make_unique<CAIF_DeviceRMSNorm>(config.dim,
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
      network.AddLayer(
          std::make_unique<CAIF_DeviceLinearHead>(
              head_cfg,
              emb_layer.ParameterTensor(0),
              emb_layer.GradientTensor(0),
              stream));
    }
    else
    {
      network.AddLayer(
          std::make_unique<CAIF_DeviceLinearHead>(head_cfg,stream));
    }

    ISE_Out::Out()<<"Qwen model built: "
                  <<network.LayerCount()
                  <<" layers, "
                  <<network.TotalParameterCount()
                  <<" total parameters"
                  <<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_QwenModelBuilder::BuildModel")
}

//------------------------------------------------------------------------------
// LoadWeights
//------------------------------------------------------------------------------
void RTNR_QwenModelBuilder::LoadWeights(
                                 CAIF_DeviceNetwork &network,
                                 CAIF_CudaStream &stream,
                                 const std::string &model_dir,
                                 const RTNR_QwenConfig_t &config,
                                 CAIF_DataType::CAIF_DataType_e storage_dtype)
{
  try
  {
    CAIF_SafeTensorsFormat st;

    ISE_Out::Out()<<"Loading weights from "<<model_dir<<std::endl;

    // Layer 0: Token embedding
    CAIF_DeviceTensor emb_w=st.LoadTensorByName(
                               model_dir,
                               "model.embed_tokens.weight",
                               stream);
    network.Layer(0).ParameterTensor(0)=
        emb_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
    ISE_Out::Out()<<"  Loaded embedding"<<std::endl;

    // Load all FrozenLinear projection weights via the weight map
    uint32_t proj_count=0;
    for(auto &[hf_name,frozen]:_weight_map)
    {
      CAIF_DeviceTensor raw_w=st.LoadTensorByName(model_dir,
                                                   hf_name,
                                                   stream);

      if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float32)
      {
        frozen->LoadFromTensor(
            raw_w.To(CAIF_DataType::CAIF_DataType_e::Float32));
      }
      else if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Float16 ||
              storage_dtype==CAIF_DataType::CAIF_DataType_e::BFloat16)
      {
        frozen->LoadFromTensor(raw_w.To(storage_dtype));
      }
      else if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Int8)
      {
        frozen->LoadFromTensor(
            raw_w.To(CAIF_DataType::CAIF_DataType_e::Int8));
      }
      else if(storage_dtype==CAIF_DataType::CAIF_DataType_e::Int4)
      {
        CAIF_DeviceTensor fp32_w=
            raw_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
        uint32_t rows=fp32_w.Shape()[0];
        uint32_t cols=fp32_w.Shape()[1];
        uint32_t num_groups=
            (rows*cols+g_caif_quant_default_group_size-1)/
            g_caif_quant_default_group_size;

        CAIF_DeviceTensor int4_w=
            CAIF_DeviceTensor::Uninitialized(
                {(rows*cols+1)/2},
                stream,
                CAIF_DataType::CAIF_DataType_e::UInt8);
        CAIF_DeviceTensor scales=
            CAIF_DeviceTensor::Uninitialized(
                {num_groups},
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
        frozen->LoadScalesFromHost(host_scales.data(),
                                    num_groups*2);
      }
      else
      {
        frozen->LoadFromTensor(raw_w.To(storage_dtype));
      }

      ++proj_count;
    }
    ISE_Out::Out()<<"  Loaded "
                  <<proj_count
                  <<" projection weights"
                  <<std::endl;

    // Transformer layer norms and QKV biases
    for(uint32_t layer_idx=0;layer_idx<config.num_layers;++layer_idx)
    {
      std::string prefix=
          "model.layers."+std::to_string(layer_idx)+".";
      auto &block=network.Layer(layer_idx+1);

      // Input layernorm (sublayer 0 norm gamma = parameter index 0)
      std::string norm_name=prefix+"input_layernorm.weight";
      CAIF_DeviceTensor norm_w=st.LoadTensorByName(model_dir,
                                                    norm_name,
                                                    stream);
      block.ParameterTensor(0)=
          norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);

      // Post-attention layernorm
      std::vector<std::string> names=block.ParameterNames("");
      norm_name=prefix+"post_attention_layernorm.weight";
      norm_w=st.LoadTensorByName(model_dir,norm_name,stream);

      for(size_t i=0;i<names.size();++i)
      {
        if(names[i].find("post_attention_layernorm.")!=
           std::string::npos)
        {
          block.ParameterTensor(i)=
              norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
          break;
        }
      }

      // Load QKV biases via stored MHA raw pointers
      if(config.use_qkv_bias==true)
      {
        CAIF_DeviceTensor q_b=st.LoadTensorByName(
            model_dir,
            prefix+"self_attn.q_proj.bias",
            stream);
        _mha_layers[layer_idx]->QBias()=
            q_b.To(CAIF_DataType::CAIF_DataType_e::Float32);

        CAIF_DeviceTensor k_b=st.LoadTensorByName(
            model_dir,
            prefix+"self_attn.k_proj.bias",
            stream);
        _mha_layers[layer_idx]->KBias()=
            k_b.To(CAIF_DataType::CAIF_DataType_e::Float32);

        CAIF_DeviceTensor v_b=st.LoadTensorByName(
            model_dir,
            prefix+"self_attn.v_proj.bias",
            stream);
        _mha_layers[layer_idx]->VBias()=
            v_b.To(CAIF_DataType::CAIF_DataType_e::Float32);
      }

      if((layer_idx+1)%10==0 || layer_idx==config.num_layers-1)
      {
        ISE_Out::Out()<<"  Loaded norms for layer "
                      <<(layer_idx+1)
                      <<"/"
                      <<config.num_layers
                      <<std::endl;
      }
    }

    // Final RMSNorm
    CAIF_DeviceTensor final_norm_w=st.LoadTensorByName(
                                      model_dir,
                                      "model.norm.weight",
                                      stream);
    size_t final_norm_idx=config.num_layers+1;
    network.Layer(final_norm_idx).ParameterTensor(0)=
        final_norm_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
    ISE_Out::Out()<<"  Loaded final norm"<<std::endl;

    // lm_head (if not tied)
    if(config.tie_word_embeddings==false)
    {
      CAIF_DeviceTensor head_w=st.LoadTensorByName(model_dir,
                                                    "lm_head.weight",
                                                    stream);
      CAIF_DeviceTensor head_w_fp32=
          head_w.To(CAIF_DataType::CAIF_DataType_e::Float32);
      CAIF_DeviceTensor head_w_t=
          CAIF_DeviceTensor::Uninitialized(
              {config.dim,config.vocab_size},stream);
      CAIF_DeviceOps::Transpose(head_w_fp32,head_w_t);
      size_t head_idx=config.num_layers+2;
      network.Layer(head_idx).ParameterTensor(0)=
          std::move(head_w_t);
      ISE_Out::Out()<<"  Loaded lm_head"<<std::endl;
    }

    ISE_Out::Out()<<"Weight loading complete"<<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_QwenModelBuilder::LoadWeights")
}

//------------------------------------------------------------------------------
// SaveLoRAWeights
//------------------------------------------------------------------------------
void RTNR_QwenModelBuilder::SaveLoRAWeights(
                                 const CAIF_DeviceNetwork &network,
                                 const std::string &path)
{
  try
  {
    std::vector<std::pair<std::string,const CAIF_DeviceTensor*>> tensors;

    // Iterate all layers, collect LoRA parameters
    for(size_t layer_idx=0;layer_idx<network.LayerCount();++layer_idx)
    {
      const auto &layer=network.Layer(layer_idx);
      std::string layer_prefix=
          "layers."+std::to_string(layer_idx)+".";

      std::vector<std::string> names=
          layer.ParameterNames(layer_prefix);

      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        // LoRA parameters have "lora_a" or "lora_b" in their names
        if(names[p].find("lora_a")!=std::string::npos ||
           names[p].find("lora_b")!=std::string::npos)
        {
          tensors.push_back({names[p],&layer.ParameterTensor(p)});
        }
      }
    }

    if(tensors.empty()==true)
    {
      ISE_Out::Out()<<"No LoRA parameters found to save"
                    <<std::endl;
      return;
    }

    std::map<std::string,std::string> metadata;
    metadata["format"]="lora";
    metadata["framework"]="caif";

    CAIF_SafeTensorsFormat st;
    st.Save(path,tensors,metadata);

    ISE_Out::Out()<<"Saved "
                  <<tensors.size()
                  <<" LoRA tensors to "
                  <<path
                  <<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_QwenModelBuilder::SaveLoRAWeights")
}

//------------------------------------------------------------------------------
// LoadLoRAWeights
//------------------------------------------------------------------------------
void RTNR_QwenModelBuilder::LoadLoRAWeights(
                                 CAIF_DeviceNetwork &network,
                                 const std::string &path,
                                 CAIF_CudaStream &stream)
{
  try
  {
    CAIF_SafeTensorsFormat st;
    std::map<std::string,CAIF_DeviceTensor> loaded=
        st.Load(path,stream);

    uint32_t loaded_count=0;

    // Match loaded tensors to network parameters by name
    for(size_t layer_idx=0;layer_idx<network.LayerCount();++layer_idx)
    {
      auto &layer=network.Layer(layer_idx);
      std::string layer_prefix=
          "layers."+std::to_string(layer_idx)+".";

      std::vector<std::string> names=
          layer.ParameterNames(layer_prefix);

      for(size_t p=0;p<layer.ParameterTensorCount();++p)
      {
        auto it=loaded.find(names[p]);
        if(it!=loaded.end())
        {
          CAIF_DeviceTensor fp32=
              it->second.To(CAIF_DataType::CAIF_DataType_e::Float32);
          layer.ParameterTensor(p)=std::move(fp32);
          ++loaded_count;
        }
      }
    }

    ISE_Out::Out()<<"Loaded "
                  <<loaded_count
                  <<" LoRA tensors from "
                  <<path
                  <<std::endl;
  }
  RTNR_CATCH_BLOCK("RTNR_QwenModelBuilder::LoadLoRAWeights")
}
