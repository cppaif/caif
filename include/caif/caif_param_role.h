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

#pragma once

#include "caif_base.h"
#include <cstdint>

namespace instance
{

// Scoping class for the three enums caif uses to classify every
// trainable / frozen parameter the layer system can emit. The three
// enums are intentionally orthogonal:
//
//   Role_e   — the specific role (1-of-N exhaustive list)
//   Family_e — the broad layer family the role belongs to
//   Kind_e   — the kind of tensor (weight, bias, gamma, etc.)
//
// Roles are listed contiguously starting at Unknown_e=0 and ending at
// COUNT_e. CAIF_RoleRegistry indexes a vector by static_cast<size_t>
// for O(1) lookup; new roles are appended just before COUNT_e.
class CAIF_ParamRole:public CAIF_Base
{
  public:
    enum class Role_e:uint16_t
    {
      Unknown_e=0,

      // Attention (MHA / GQA / Cross)
      AttnWQ_e,
      AttnWK_e,
      AttnWV_e,
      AttnWO_e,
      AttnBiasQ_e,
      AttnBiasK_e,
      AttnBiasV_e,
      AttnQNormGamma_e,
      AttnKNormGamma_e,

      // Multi-head Latent Attention
      MLAWQCompress_e,
      MLAWQDecompress_e,
      MLAWKVCompress_e,
      MLAWKVDecompress_e,
      MLAWO_e,
      MLAQNormGamma_e,
      MLAKVNormGamma_e,

      // FFN
      FFNWGate_e,
      FFNWUp_e,
      FFNWDown_e,
      FFNBiasGate_e,
      FFNBiasUp_e,
      FFNBiasDown_e,

      // MoE
      MoERouterWeight_e,
      MoERouterBias_e,
      MoEExpertGate_e,
      MoEExpertUp_e,
      MoEExpertDown_e,
      MoESharedExpertGate_e,
      MoESharedExpertUp_e,
      MoESharedExpertDown_e,
      MoEExpertBiasGate_e,
      MoEExpertBiasUp_e,
      MoEExpertBiasDown_e,

      // Norms
      RMSNormGamma_e,
      LayerNormGamma_e,
      LayerNormBeta_e,
      FinalNormGamma_e,

      // Embedding tables
      TokenEmbeddingTable_e,
      PositionEmbeddingTable_e,

      // Heads
      LinearHeadWeight_e,
      LinearHeadBias_e,

      // LoRA adapters
      LoRA_A_e,
      LoRA_B_e,

      // Vision (conv + BN)
      ConvWeight_e,
      ConvBias_e,
      BNGamma_e,
      BNBeta_e,
      BNRunningMean_e,
      BNRunningVar_e,

      // T5 relative position bias
      RelativePositionBias_e,

      // Generic embedding projection (patch / tabular / spectrogram).
      // Shared across all three input-embedding flavors — disambiguation
      // is structural (the container's path tells you which embedding
      // type owns this slot).
      EmbedProjWeight_e,
      EmbedProjBias_e,
      EmbedClsToken_e,

      // Generic terminal weight / bias for layers whose semantic role is
      // determined by the container's structural prefix (DenseLayer,
      // Conv2D, FrozenLinear). Disambiguation is structural; the leaf
      // name itself carries no semantics.
      GenericWeight_e,
      GenericBias_e,

      // Structural path segments — emitted by container layers as the
      // prefix that addresses a child slot. Like leaf roles, these are
      // overridable via CAIF_RoleRegistry::SetName so callers can fully
      // control every externally-visible string caif emits.
      PathMoERouter_e,
      PathMoEExpert_e,
      PathMoESharedExpert_e,
      PathTransformerBlocks_e,
      PathViTBlocks_e,
      PathEmbedIn_e,
      PathEmbedPos_e,
      PathFinalNorm_e,
      PathHead_e,
      PathAttnNorm_e,
      PathAttn_e,
      PathFFNNorm_e,
      PathFFN_e,
      PathViTPatchEmbed_e,
      PathGenericContainerLayer_e,

      // Sentinel — must remain the last value.
      COUNT_e
    };

    enum class Family_e:uint8_t
    {
      Unknown_e=0,
      Attention_e,
      MLA_e,
      FFN_e,
      MoE_e,
      Norm_e,
      Embedding_e,
      Head_e,
      LoRA_e,
      Vision_e,
      PositionBias_e,
      StructuralPath_e
    };

    enum class Kind_e:uint8_t
    {
      Unknown_e=0,
      Weight_e,
      Bias_e,
      Gamma_e,
      Beta_e,
      RunningStat_e,
      Adapter_e,
      Table_e,
      PathSegment_e
    };

  protected:

  private:
    CAIF_ParamRole()=delete;
    ~CAIF_ParamRole()override=default;
    CAIF_ParamRole(const CAIF_ParamRole &)=delete;
    CAIF_ParamRole &operator=(const CAIF_ParamRole &)=delete;
};

}//end instance namespace
