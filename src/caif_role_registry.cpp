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

#include "caif_role_registry.h"
#include "caif_constants.h"
#include "caif_exception.h"

namespace instance
{

CAIF_RoleRegistry &CAIF_RoleRegistry::Instance()
{
  static CAIF_RoleRegistry s_instance;
  return s_instance;
}

CAIF_RoleRegistry::CAIF_RoleRegistry():_info_by_role(),
                                      _by_family(),
                                      _by_kind(),
                                      _by_name()
{
  try
  {
    PopulateCanonicalTable();
    RebuildIndexes();
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_RoleRegistry::SetRow(const CAIF_ParamRole::Role_e role,
                               const std::string &name,
                               const CAIF_ParamRole::Family_e family,
                               const CAIF_ParamRole::Kind_e kind,
                               const bool default_trainable,
                               const CAIF_DataType::CAIF_DataType_e default_dtype)
{
  try
  {
    const size_t idx=static_cast<size_t>(role);
    if(idx>=InfoVec().size())
    {
      THROW_CAIFE("CAIF_RoleRegistry::SetRow: role index out of range");
    }
    InfoVecMut()[idx]=CAIF_RoleInfo(role,
                                    name,
                                    family,
                                    kind,
                                    default_trainable,
                                    default_dtype);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_RoleRegistry::PopulateCanonicalTable()
{
  try
  {
    InfoVecMut().resize(static_cast<size_t>(CAIF_ParamRole::Role_e::COUNT_e));

    // Attention (MHA / GQA / Cross)
    SetRow(CAIF_ParamRole::Role_e::AttnWQ_e,
           g_caif_role_name_attn_w_q,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::AttnWK_e,
           g_caif_role_name_attn_w_k,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::AttnWV_e,
           g_caif_role_name_attn_w_v,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::AttnWO_e,
           g_caif_role_name_attn_w_o,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::AttnBiasQ_e,
           g_caif_role_name_attn_bias_q,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::AttnBiasK_e,
           g_caif_role_name_attn_bias_k,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::AttnBiasV_e,
           g_caif_role_name_attn_bias_v,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::AttnQNormGamma_e,
           g_caif_role_name_attn_q_norm_gamma,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Gamma_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::AttnKNormGamma_e,
           g_caif_role_name_attn_k_norm_gamma,
           CAIF_ParamRole::Family_e::Attention_e,
           CAIF_ParamRole::Kind_e::Gamma_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // Multi-head Latent Attention
    SetRow(CAIF_ParamRole::Role_e::MLAWQCompress_e,
           g_caif_role_name_mla_w_q_compress,
           CAIF_ParamRole::Family_e::MLA_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MLAWQDecompress_e,
           g_caif_role_name_mla_w_q_decompress,
           CAIF_ParamRole::Family_e::MLA_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MLAWKVCompress_e,
           g_caif_role_name_mla_w_kv_compress,
           CAIF_ParamRole::Family_e::MLA_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MLAWKVDecompress_e,
           g_caif_role_name_mla_w_kv_decompress,
           CAIF_ParamRole::Family_e::MLA_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MLAWO_e,
           g_caif_role_name_mla_w_o,
           CAIF_ParamRole::Family_e::MLA_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MLAQNormGamma_e,
           g_caif_role_name_mla_q_norm_gamma,
           CAIF_ParamRole::Family_e::MLA_e,
           CAIF_ParamRole::Kind_e::Gamma_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MLAKVNormGamma_e,
           g_caif_role_name_mla_kv_norm_gamma,
           CAIF_ParamRole::Family_e::MLA_e,
           CAIF_ParamRole::Kind_e::Gamma_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // FFN
    SetRow(CAIF_ParamRole::Role_e::FFNWGate_e,
           g_caif_role_name_ffn_w_gate,
           CAIF_ParamRole::Family_e::FFN_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::FFNWUp_e,
           g_caif_role_name_ffn_w_up,
           CAIF_ParamRole::Family_e::FFN_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::FFNWDown_e,
           g_caif_role_name_ffn_w_down,
           CAIF_ParamRole::Family_e::FFN_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::FFNBiasGate_e,
           g_caif_role_name_ffn_bias_gate,
           CAIF_ParamRole::Family_e::FFN_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::FFNBiasUp_e,
           g_caif_role_name_ffn_bias_up,
           CAIF_ParamRole::Family_e::FFN_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::FFNBiasDown_e,
           g_caif_role_name_ffn_bias_down,
           CAIF_ParamRole::Family_e::FFN_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // MoE
    SetRow(CAIF_ParamRole::Role_e::MoERouterWeight_e,
           g_caif_role_name_moe_router_w,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoERouterBias_e,
           g_caif_role_name_moe_router_bias,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoEExpertGate_e,
           g_caif_role_name_moe_expert_w_gate,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoEExpertUp_e,
           g_caif_role_name_moe_expert_w_up,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoEExpertDown_e,
           g_caif_role_name_moe_expert_w_down,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoESharedExpertGate_e,
           g_caif_role_name_moe_shared_expert_w_gate,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoESharedExpertUp_e,
           g_caif_role_name_moe_shared_expert_w_up,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoESharedExpertDown_e,
           g_caif_role_name_moe_shared_expert_w_down,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoEExpertBiasGate_e,
           g_caif_role_name_moe_expert_b_gate,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoEExpertBiasUp_e,
           g_caif_role_name_moe_expert_b_up,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::MoEExpertBiasDown_e,
           g_caif_role_name_moe_expert_b_down,
           CAIF_ParamRole::Family_e::MoE_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // Norms
    SetRow(CAIF_ParamRole::Role_e::RMSNormGamma_e,
           g_caif_role_name_rmsnorm_gamma,
           CAIF_ParamRole::Family_e::Norm_e,
           CAIF_ParamRole::Kind_e::Gamma_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::LayerNormGamma_e,
           g_caif_role_name_layernorm_gamma,
           CAIF_ParamRole::Family_e::Norm_e,
           CAIF_ParamRole::Kind_e::Gamma_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::LayerNormBeta_e,
           g_caif_role_name_layernorm_beta,
           CAIF_ParamRole::Family_e::Norm_e,
           CAIF_ParamRole::Kind_e::Beta_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::FinalNormGamma_e,
           g_caif_role_name_final_norm_gamma,
           CAIF_ParamRole::Family_e::Norm_e,
           CAIF_ParamRole::Kind_e::Gamma_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // Embedding tables
    SetRow(CAIF_ParamRole::Role_e::TokenEmbeddingTable_e,
           g_caif_role_name_token_embedding_table,
           CAIF_ParamRole::Family_e::Embedding_e,
           CAIF_ParamRole::Kind_e::Table_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PositionEmbeddingTable_e,
           g_caif_role_name_position_embedding_table,
           CAIF_ParamRole::Family_e::Embedding_e,
           CAIF_ParamRole::Kind_e::Table_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // Heads
    SetRow(CAIF_ParamRole::Role_e::LinearHeadWeight_e,
           g_caif_role_name_linear_head_w,
           CAIF_ParamRole::Family_e::Head_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::LinearHeadBias_e,
           g_caif_role_name_linear_head_bias,
           CAIF_ParamRole::Family_e::Head_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // LoRA adapters
    SetRow(CAIF_ParamRole::Role_e::LoRA_A_e,
           g_caif_role_name_lora_a,
           CAIF_ParamRole::Family_e::LoRA_e,
           CAIF_ParamRole::Kind_e::Adapter_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::LoRA_B_e,
           g_caif_role_name_lora_b,
           CAIF_ParamRole::Family_e::LoRA_e,
           CAIF_ParamRole::Kind_e::Adapter_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // Vision (conv + BN)
    SetRow(CAIF_ParamRole::Role_e::ConvWeight_e,
           g_caif_role_name_conv_w,
           CAIF_ParamRole::Family_e::Vision_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::ConvBias_e,
           g_caif_role_name_conv_bias,
           CAIF_ParamRole::Family_e::Vision_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::BNGamma_e,
           g_caif_role_name_bn_gamma,
           CAIF_ParamRole::Family_e::Vision_e,
           CAIF_ParamRole::Kind_e::Gamma_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::BNBeta_e,
           g_caif_role_name_bn_beta,
           CAIF_ParamRole::Family_e::Vision_e,
           CAIF_ParamRole::Kind_e::Beta_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::BNRunningMean_e,
           g_caif_role_name_bn_running_mean,
           CAIF_ParamRole::Family_e::Vision_e,
           CAIF_ParamRole::Kind_e::RunningStat_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::BNRunningVar_e,
           g_caif_role_name_bn_running_var,
           CAIF_ParamRole::Family_e::Vision_e,
           CAIF_ParamRole::Kind_e::RunningStat_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // T5 relative position bias
    SetRow(CAIF_ParamRole::Role_e::RelativePositionBias_e,
           g_caif_role_name_relative_position_bias,
           CAIF_ParamRole::Family_e::PositionBias_e,
           CAIF_ParamRole::Kind_e::Table_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // Shared input-embedding projection (patch / tabular / spectrogram).
    SetRow(CAIF_ParamRole::Role_e::EmbedProjWeight_e,
           g_caif_role_name_embed_proj_w,
           CAIF_ParamRole::Family_e::Embedding_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::EmbedProjBias_e,
           g_caif_role_name_embed_proj_bias,
           CAIF_ParamRole::Family_e::Embedding_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::EmbedClsToken_e,
           g_caif_role_name_embed_cls_token,
           CAIF_ParamRole::Family_e::Embedding_e,
           CAIF_ParamRole::Kind_e::Table_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // Generic terminal weight / bias (DenseLayer, Conv2D, FrozenLinear).
    SetRow(CAIF_ParamRole::Role_e::GenericWeight_e,
           g_caif_role_name_generic_weight,
           CAIF_ParamRole::Family_e::Unknown_e,
           CAIF_ParamRole::Kind_e::Weight_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::GenericBias_e,
           g_caif_role_name_generic_bias,
           CAIF_ParamRole::Family_e::Unknown_e,
           CAIF_ParamRole::Kind_e::Bias_e,
           true,
           CAIF_DataType::CAIF_DataType_e::Float32);

    // Structural path segments. Default values match what caif emitted
    // before path segments became overridable; trailing "." / "_"
    // remain in the value because the layer's `prefix + reg.Name(...) +
    // suffix` composition expects them in the role's literal string.
    SetRow(CAIF_ParamRole::Role_e::PathMoERouter_e,
           g_caif_path_moe_router,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathMoEExpert_e,
           g_caif_path_moe_expert,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathMoESharedExpert_e,
           g_caif_path_moe_shared_expert,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathTransformerBlocks_e,
           g_caif_path_transformer_blocks,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathViTBlocks_e,
           g_caif_path_vit_blocks,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathEmbedIn_e,
           g_caif_path_embed_in,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathEmbedPos_e,
           g_caif_path_embed_pos,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathFinalNorm_e,
           g_caif_path_final_norm,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathHead_e,
           g_caif_path_head,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathAttnNorm_e,
           g_caif_path_attn_norm,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathAttn_e,
           g_caif_path_attn,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathFFNNorm_e,
           g_caif_path_ffn_norm,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathFFN_e,
           g_caif_path_ffn,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathViTPatchEmbed_e,
           g_caif_path_vit_patch_embed,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
    SetRow(CAIF_ParamRole::Role_e::PathGenericContainerLayer_e,
           g_caif_path_generic_container_layer,
           CAIF_ParamRole::Family_e::StructuralPath_e,
           CAIF_ParamRole::Kind_e::PathSegment_e,
           false,
           CAIF_DataType::CAIF_DataType_e::Float32);
  }
  CAIF_CATCH_BLOCK();
}

void CAIF_RoleRegistry::RebuildIndexes()
{
  try
  {
    ByFamilyMut().clear();
    ByKindMut().clear();
    ByNameMut().clear();

    for(size_t i=0;i<InfoVec().size();++i)
    {
      const CAIF_RoleInfo &row=InfoVec()[i];
      if(row.Role()==CAIF_ParamRole::Role_e::Unknown_e)
      {
        continue;
      }
      ByFamilyMut()[row.Family()].push_back(row.Role());
      ByKindMut()[row.Kind()].push_back(row.Role());
      if(row.Name().empty()==false)
      {
        ByNameMut()[row.Name()]=row.Role();
      }
    }
  }
  CAIF_CATCH_BLOCK();
}

const CAIF_RoleInfo &CAIF_RoleRegistry::Info(const CAIF_ParamRole::Role_e role)const
{
  try
  {
    const size_t idx=static_cast<size_t>(role);
    if(idx>=InfoVec().size())
    {
      THROW_CAIFE("CAIF_RoleRegistry::Info: role index out of range");
    }
    return InfoVec()[idx];
  }
  CAIF_CATCH_BLOCK();
}

const std::string &CAIF_RoleRegistry::Name(const CAIF_ParamRole::Role_e role)const
{
  return Info(role).Name();
}

CAIF_ParamRole::Family_e CAIF_RoleRegistry::Family(const CAIF_ParamRole::Role_e role)const
{
  return Info(role).Family();
}

CAIF_ParamRole::Kind_e CAIF_RoleRegistry::Kind(const CAIF_ParamRole::Role_e role)const
{
  return Info(role).Kind();
}

bool CAIF_RoleRegistry::DefaultTrainable(const CAIF_ParamRole::Role_e role)const
{
  return Info(role).DefaultTrainable();
}

CAIF_DataType::CAIF_DataType_e CAIF_RoleRegistry::DefaultDtype(const CAIF_ParamRole::Role_e role)const
{
  return Info(role).DefaultDtype();
}

const CAIF_RoleRegistry::RoleVec_t &CAIF_RoleRegistry::RolesByFamily(const CAIF_ParamRole::Family_e family)const
{
  try
  {
    FamilyToRolesMap_t::const_iterator it=ByFamily().find(family);
    if(it==ByFamily().end())
    {
      THROW_CAIFE("CAIF_RoleRegistry::RolesByFamily: family not present");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

const CAIF_RoleRegistry::RoleVec_t &CAIF_RoleRegistry::RolesByKind(const CAIF_ParamRole::Kind_e kind)const
{
  try
  {
    KindToRolesMap_t::const_iterator it=ByKind().find(kind);
    if(it==ByKind().end())
    {
      THROW_CAIFE("CAIF_RoleRegistry::RolesByKind: kind not present");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

CAIF_ParamRole::Role_e CAIF_RoleRegistry::RoleByName(const std::string &name)const
{
  try
  {
    NameToRoleMap_t::const_iterator it=ByName().find(name);
    if(it==ByName().end())
    {
      THROW_CAIFE("CAIF_RoleRegistry::RoleByName: name not registered");
    }
    return it->second;
  }
  CAIF_CATCH_BLOCK();
}

bool CAIF_RoleRegistry::HasRoleForName(const std::string &name)const
{
  return ByName().find(name)!=ByName().end();
}

void CAIF_RoleRegistry::SetName(const CAIF_ParamRole::Role_e role,const std::string &name)
{
  try
  {
    const size_t idx=static_cast<size_t>(role);
    if(idx>=InfoVec().size())
    {
      THROW_CAIFE("CAIF_RoleRegistry::SetName: role index out of range");
    }
    InfoVecMut()[idx].SetName(name);
    RebuildIndexes();
  }
  CAIF_CATCH_BLOCK();
}

}//end instance namespace
